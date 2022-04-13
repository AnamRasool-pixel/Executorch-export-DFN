from abc import ABC, abstractmethod
from functools import partial
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiFrameModule(nn.Module, ABC):
    """Multi-frame speech enhancement modules.

    Signal model and notation:
        Noisy: `x = s + n`
        Enhanced: `y = f(x)`
        Objective: `min ||s - y||`

        PSD: Power spectral density, notated eg. as `Rxx` for noisy PSD.
        IFC: Inter-frame correlation vector: PSD*u, u: selection vector. Notated as `rxx`
    """

    num_freqs: Final[int]
    frame_size: Final[int]
    need_pad: Final[bool]

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.pad = nn.ConstantPad2d((0, 0, frame_size - lookahead - 1, lookahead), 0.0)
        self.need_pad = frame_size > 1 or lookahead != 0

    def pad_unfold(self, spec: Tensor):
        """Pads and unfolds the spectrogram according to frame_size.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C, T, F]
        Returns:
            spec (Tensor): Unfolded spectrogram of shape [B, C, T, F, N], where N: frame_size.
        """
        if self.need_pad:
            return self.pad(spec).unfold(2, self.frame_size, 1)
        return spec.unsqueeze(-1)

    def forward(self, spec: Tensor, coefs: Tensor):
        """Pads and unfolds the spectrogram and forwards to impl.

        Args:
            spec (Tensor): Spectrogram of shape [B, C, T, F, 2]
            coefs (Tensor): Spectrogram of shape [B, C, T, F, 2]
        """
        spec_u = self.pad_unfold(torch.view_as_complex(spec))
        coefs = torch.view_as_complex(coefs)
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)
        spec_f = self.forward_impl(spec_f, coefs)
        if self.training:
            spec = spec.clone()
        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec

    @abstractmethod
    def forward_impl(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """Forward impl taking complex spectrogram and coefficients.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C1, T, F, N]
            coefs (complex Tensor): Coefficients [B, C2, T, F]

        Returns:
            spec (complex Tensor): Enhanced spectrogram of shape [B, C1, T, F]
        """
        ...

    @abstractmethod
    def num_channels(self) -> int:
        """Return the number of required channels.

        If multiple inputs are required, then all these should be combined in one Tensor containing
        the summed channels.
        """
        ...


def psd(x: Tensor, n: int) -> Tensor:
    """Compute the PSD correlation matrix Rxx for a spectrogram.

    That is, `X*conj(X)`, where `*` is the outer product.

    Args:
        x (complex Tensor): Spectrogram of shape [B, C, T, F]. Will be unfolded with `n` steps over
            the time axis.

    Returns:
        Rxx (complex Tensor): Correlation matrix of shape [B, C, T, F, N, N]
    """
    x = F.pad(x, (n - 1, 0, 0, 0)).unfold(-2, n, -1)
    return torch.einsum("...n,...m->...mn", x, x.conj())


def df(spec: Tensor, coefs: Tensor) -> Tensor:
    """Deep filter implemenation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F, N]
        coefs (complex Tensor): Spectrogram of shape [B, C, N, T, F]

    Returns:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F]
    """
    return torch.einsum("...tfn,...ntf->...tf", spec, coefs)


class CRM(MultiFrameModule):
    """Complex ratio mask."""

    def __init__(self, num_freqs: int):
        super().__init__(num_freqs, 1)

    def forward_impl(self, spec: Tensor, coefs: Tensor):
        return spec.mul(coefs).squeeze(-1)

    def num_channels(self):
        return 2


class DF(MultiFrameModule):
    conj: Final[bool]
    """Deep Filtering."""

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0, conj: bool = False):
        super().__init__(num_freqs, frame_size, lookahead)
        self.conj = conj

    def forward_impl(self, spec: Tensor, coefs: Tensor):
        coefs = coefs.unflatten(1, (-1, self.frame_size))
        if self.conj:
            coefs = coefs.conj()
        return df(spec, coefs)

    def num_channels(self):
        return self.frame_size * 2


class MfWf(MultiFrameModule):
    """Multi-frame Wiener Filter."""

    def __init__(
        self, num_freqs: int, frame_size: int, lookahead: int = 0, method: str = "psd_ifc"
    ):
        """Multi-frame Wiener Filter.

        Several implementation methods are available resulting in different number of required input
        coefficient channels.

        Methods:
            psd_ifc: Predict PSD `Rxx` and IFC `rss`.
            df: Use deep filtering to predict speech and noisy spectrograms. These will be used for
                PSD calculation for Wiener filtering. Alias: `df_sx`
            c: Directly predict Wiener filter coefficients. Computation same as deep filtering.

        """
        super().__init__(num_freqs, frame_size, lookahead=0)
        self.idx = -lookahead
        self.method = method.lower()
        methods = {
            "df": (self.mfwf_dfsx, frame_size * 4),  # 2x deep filter
            "df_sx": (self.mfwf_dfsx, frame_size * 4),  # 2x deep filter
            "psd_ifc": (self.mfwf_psd_ifc, frame_size**2 + frame_size),  # Rxx+rss
            "c": (self.mfwf_c, frame_size * 2),
        }
        assert self.method in methods
        self.forward_impl, self._num_channels = methods[self.method]

    def num_channels(self):
        return self.num_channels

    @staticmethod
    def _mfwf(Rxx, rss) -> Tensor:
        return torch.einsum("...nm,...m->...n", torch.inverse(Rxx), rss)  # [T, F, N]

    def mfwf_dfsx(self, spec: Tensor, coefs: Tensor) -> Tensor:
        df_s, df_x = torch.split(coefs, 2, 1)  # [B, C, T, F, N]
        df_s = df_s.unflatten(1, (-1, self.frame_size))
        df_x = df_x.unflatten(1, (-1, self.frame_size))
        spec_s = df(spec, df_s)  # [B, C, T, F]
        spec_x = df(spec, df_x)
        Rss = psd(spec_s, self.frame_size)  # [B, C, T, F, N. N]
        Rxx = psd(spec_x, self.frame_size)
        rss = Rss[-1]  # TODO: use -1 or self.idx?
        c = self._mfwf(Rxx, rss)  # [B, C, T, F, N]
        return self.apply_coefs(spec, c)

    def mfwf_psd_ifc(self, spec: Tensor, coefs: Tensor) -> Tensor:
        Rxx, rss = torch.split(coefs, [self.frame_size**2, self.frame_size], 1)
        c = self._mfwf(Rxx, rss)
        return self.apply_coefs(spec, c)

    def mfwf_c(self, spec: Tensor, coefs: Tensor) -> Tensor:
        coefs = coefs.unflatten(1, (-1, self.frame_size)).permute(
            0, 1, 3, 4, 2
        )  # [B, C*N, T, F] -> [B, C, T, F, N]
        return self.apply_coefs(spec, coefs)

    def apply_coefs(self, spec: Tensor, coefs: Tensor) -> Tensor:
        # spec: [B, C, T, F, N]
        # coefs: [B, C, T, F, N]
        return torch.einsum("...n,...n->...", spec, coefs)


class MvdrSouden(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


class MvdrEvd(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


class MvdrRtfPower(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


MF_METHODS = {
    "crm": CRM,
    "df": DF,
    "mfwf_df": partial(MfWf, method="df"),
    "mfwf_psd": partial(MfWf, method="psd_ifc"),
    "mfwf_c": partial(MfWf, method="c"),
}
