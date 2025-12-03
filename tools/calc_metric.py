import os
from glob import glob

import torch

from metrics import basicMetric as M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Metrics:
    def __init__(self, metrics=[], skip_ref_frames=2, batch_size=1) -> None:
        for metric in metrics:
            if metric == "flolpips":
                print("Building FLOLPIPS metric...")
                from metrics.flolpips import calc_flolpips

                self.FloLPIPS = calc_flolpips
            elif metric == "vfips":
                print("Building VFIPS metric...")
                from metrics.VFIPS import calc_vfips

                self.VFIPS = calc_vfips

            elif metric in ["ssim", "msssim", "psnr", "lpips", "ie", "dists"]:
                self.kwargs = dict(device=DEVICE, verbose=0, batch=batch_size)
            else:
                raise ValueError("metric %s is not supported." % metric)

        self.metrics = metrics
        self.skip = skip_ref_frames

    def _filter_ref_frames_(self, images: list):
        if isinstance(self.skip, int):
            del images[:: self.skip]

    def calc_flolpips(self, meta):
        tmpDir = meta["tmpDir"]
        disMP4 = meta["disMP4"]
        refMP4 = meta["refMP4"]
        hwt = meta["hwt"]
        score = self.FloLPIPS(disMP4, refMP4, tmpDir, hwt)
        return score

    def calc_vfips(self, meta):
        dis = meta["disImgs"]
        ref = meta["refImgs"]
        score = self.VFIPS(dis_dir=dis, ref_dir=ref)
        return abs(score)

    def calc_psnr(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_psnr(disImages, refImages, **self.kwargs)

    def calc_ie(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_ie(disImages, refImages, **self.kwargs)

    def calc_ssim(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_ssim(disImages, refImages, **self.kwargs)

    def calc_msssim(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_msssim(disImages, refImages, **self.kwargs)

    def calc_lpips(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_lpips(disImages, refImages, **self.kwargs)

    def calc_dists(self, meta):
        disImages = sorted(glob(os.path.join(meta["disImgs"], "*.png")))
        refImages = sorted(glob(os.path.join(meta["refImgs"], "*.png")))
        self._filter_ref_frames_(disImages)
        self._filter_ref_frames_(refImages)

        return M.calc_dists(disImages, refImages, **self.kwargs)

    def eval(self, meta):
        result = {}
        for metric in self.metrics:
            result[metric] = self.__getattribute__("calc_%s" % metric)(meta)
        return result
