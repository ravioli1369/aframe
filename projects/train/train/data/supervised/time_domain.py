import torch

from train.data.supervised.supervised import SupervisedAframeDataset

from .gwanalyzer import GWAnalyzer


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)
        detector1 = X[:, 0, :]
        detector2 = X[:, 1, :]
        gwana = GWAnalyzer(detector1.cpu().numpy())
        gwana.obtain_topological_features(True, True)
        features1 = torch.tensor(gwana.topological_features, device=X.device)
        gwana = GWAnalyzer(detector2.cpu().numpy())
        gwana.obtain_topological_features(True, True)
        features2 = torch.tensor(gwana.topological_features, device=X.device)
        X = torch.stack([features1, features2], dim=1)
        return X, y
