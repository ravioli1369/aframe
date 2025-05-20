import os
import time

import gudhi
import librosa
import numpy as np
from gtda.diagrams import Amplitude, NumberOfPoints, PersistenceEntropy, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding
from sklearn.decomposition import PCA
from sklearn.pipeline import make_union
from sklearn.utils import gen_batches
from tqdm.notebook import tqdm


class GWAnalyzer(object):
    """docstring for GWAnalyzer"""

    SAMPLE_RATE = 4096

    def __init__(self, X):
        super(GWAnalyzer, self).__init__()
        self.data = X
        self.topological_features = None
        self.spectrograms = None
        self.save_spectrograms = False

    def spectogram_features(self):
        print("start processing spectrograms")

        def gudhi_lower_star_from_vertices(image):
            """Instantiate a 2d, lower-star filtration from an image.
            Create a bigger complex, with values from `image' as 0-cubes.
            Propagate those values to edges (horizontal_, vertical_) and to
            2_cubes.
            This works thanks to assertions:
                - the values of 0-cubes in horizontal_ and vertical_ stay the
                same, since we take the maximum of the repeated values.
                - the rows without 2-cubes stay the same in center_.
            Params:
                image: numpy.ndarray, dim 2,
            Returns:
                complex_: lower star filtration from image,
                center_: array, with the lower-star filtration"""
            v = np.repeat(
                np.repeat(image, 2, axis=0), 2, axis=1
            )  # expand, mimicking the datastructure
            dimensions = [
                2 * (s - 1) + 1 for s in image.shape
            ]  # number of simplices to define in each dimension
            horizontal_ = np.maximum(
                v[:, 0:-1], v[:, 1:]
            )  # filtration values for horizontal edges (and 0-cubes)
            vertical_ = np.maximum(
                v[0:-1, :], v[1:, :]
            )  # filtration values for horizontal edges ( and 1-cubes)
            center_ = np.maximum(
                np.maximum(horizontal_[0:-1, :], horizontal_[1:, :]),
                np.maximum(vertical_[:, 0:-1], vertical_[:, 1:]),
            )  # filtration values for 2-cubes
            # rows with pair indices stay the same, f.ex. horizontal_[0] ==  np.maximum(horizontal_[0:-1,:], horizontal_[1:,:])[0]
            complex_ = gudhi.CubicalComplex(
                dimensions=dimensions,
                top_dimensional_cells=np.ravel(center_.transpose()),
            )  # transpose, due to the order of simplices
            return complex_, center_

        def compute_ph(image, super_level):
            """Compute ph, with the filtration given by the image.
            Params:
                image: np.ndarray, representing the filtration,
                f.ex. np.array([[2,0],[1,3],[4,5]]),
                super_level: boolean
            """
            tmp_image = -image if super_level else image
            complex, _ = gudhi_lower_star_from_vertices(tmp_image)
            complex.persistence(homology_coeff_field=2, min_persistence=0)
            return {"complex": complex, "super_level": super_level}

        def get_gudhi_persistence_diagram(arr):
            complex_dict = compute_ph(arr, True)
            ph = complex_dict["complex"].persistence(
                homology_coeff_field=2, min_persistence=0
            )
            return ph

        def _postprocess_diagrams(
            Xt, homology_dimensions=[0, 1], infinity_values=None, reduced=True
        ):
            # NOTE: `homology_dimensions` must be sorted in ascending order
            def replace_infinity_values(subdiagram):
                np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
                return subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]

            # Replace np.inf with infinity_values and turn into list of
            # dictionaries whose keys are the dimensions

            slices = {
                dim: slice(None) if (dim or not reduced) else slice(1, None)
                for dim in homology_dimensions
            }
            Xt = [
                {
                    dim: replace_infinity_values(
                        np.array(
                            [
                                pers_info[1]
                                for pers_info in diagram
                                if pers_info[0] == dim
                            ]
                        ).reshape(-1, 2)[slices[dim]]
                    )
                    for dim in homology_dimensions
                }
                for diagram in Xt
            ]

            # Conversion to array of triples with padding triples
            start_idx_per_dim = np.cumsum(
                [0]
                + [
                    np.max([len(diagram[dim]) for diagram in Xt] + [1])
                    for dim in homology_dimensions
                ]
            )
            min_values = [
                min(
                    [
                        np.min(diagram[dim][:, 0])
                        if diagram[dim].size
                        else np.inf
                        for diagram in Xt
                    ]
                )
                for dim in homology_dimensions
            ]
            min_values = [
                min_value if min_value != np.inf else 0
                for min_value in min_values
            ]
            n_features = start_idx_per_dim[-1]
            Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

            for i, dim in enumerate(homology_dimensions):
                start_idx, end_idx = start_idx_per_dim[i : i + 2]
                padding_value = min_values[i]
                # Add dimension as the third element of each (b, d) tuple globally
                Xt_padded[:, start_idx:end_idx, 2] = dim
                for j, diagram in enumerate(Xt):
                    subdiagram = diagram[dim]
                    end_idx_nontrivial = start_idx + len(subdiagram)
                    # Populate nontrivial part of the subdiagram
                    Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
                    # Insert padding triples
                    Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [
                        padding_value
                    ] * 2

            return Xt_padded

        if self.data is None:
            raise Exception("Empty Data")

        scaling = Scaler()

        metrics_all = [
            {"metric": metric}
            for metric in [
                "bottleneck",
                "wasserstein",
                "betti",
                "landscape",
                "silhouette",
                "heat",
                "persistence_image",
            ]
        ]

        feature_union_all = make_union(
            PersistenceEntropy(normalize=True, nan_fill_value=-10),
            NumberOfPoints(n_jobs=6),
            *[Amplitude(**metric, n_jobs=6) for metric in metrics_all],
        )

        pipe = Pipeline(
            [
                ("scaling", scaling),
                ("features", feature_union_all),
            ]
        )

        ##### Parallelize spectrogramming
        specs = []
        diagrams = []
        for wave in self.data:
            spec = librosa.stft(
                y=wave, hop_length=5, n_fft=20, window="hann", center=True
            )
            amp = np.abs(spec)
            specs.append(amp)
            diagram = get_gudhi_persistence_diagram(
                librosa.amplitude_to_db(amp)
            )
            diagrams.append(diagram)

        Xt = _postprocess_diagrams(diagrams)
        self.spectrograms = np.array(specs)

        feature_list = []
        for batch_idx in tqdm(gen_batches(Xt.shape[0], batch_size=3400)):
            chunk = Xt[batch_idx]
            print(f"Processing chunk of shape: {chunk.shape}")
            start = time.time()

            feat = pipe.fit_transform(chunk)
            feature_list.append(feat)

            end = time.time()
            print(f"Chunk Elapsed Time: {end - start}")

        return np.vstack(feature_list)

    def point_cloud_features(
        self, embedding_time_delay, embedding_dimension, stride
    ):
        # Start with Takens Embedding,
        print("start processing point cloud features")
        if self.data is None:
            raise Exception("Preprocessor not run yet")

        embedder = TakensEmbedding(
            time_delay=embedding_time_delay,
            dimension=embedding_dimension,
            stride=stride,
        )

        batch_pca = CollectionTransformer(PCA(n_components=3), n_jobs=-1)
        persistence = VietorisRipsPersistence(
            homology_dimensions=[0, 1], n_jobs=-1
        )

        scaling = Scaler()

        metrics = [
            {"metric": metric}
            for metric in [
                "bottleneck",
                "wasserstein",
                "betti",
                "landscape",
                "silhouette",
                "heat",
                "persistence_image",
            ]
        ]

        # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
        feature_union = make_union(
            PersistenceEntropy(normalize=True, nan_fill_value=-10),
            NumberOfPoints(n_jobs=-1),
            *[Amplitude(**metric, n_jobs=-1) for metric in metrics],
        )

        pipe = Pipeline(
            [
                ("embedder", embedder),
                ("pca", batch_pca),
                ("persistence2", persistence),
                ("scaling", scaling),
                ("features", feature_union),
            ]
        )

        feature_list = []
        for batch_idx in tqdm(
            gen_batches(self.data.shape[0], batch_size=3400)
        ):
            chunk = self.data[batch_idx]
            print(f"Processing chunk of shape: {chunk.shape}")
            start = time.time()

            features = pipe.fit_transform(chunk)
            feature_list.append(features)

            end = time.time()
            print(f"Chunk Elapsed Time: {end - start}")

        return np.vstack(feature_list)

    def classify(self):
        # Binary classification on Background vs Mixture
        if self.topological_features is None:
            raise Exception("Topololgical features not obtained yet")

        print("Bring Your Own Classifier")
        pass

    def obtain_topological_features(
        self, use_pointcloud: bool, use_spectogram: bool
    ):
        if self.data is None:
            raise Exception("Preprocessor not run yet")

        if not use_pointcloud and not use_spectogram:
            raise Exception("At least one has to be true")

        _features = np.empty((self.data.shape[0], 0))

        if use_spectogram:
            _features = np.hstack([_features, self.spectogram_features()])

        if use_pointcloud:
            _features = np.hstack(
                (
                    _features,
                    self.point_cloud_features(1, 10, 1).reshape(
                        self.data.shape[0], -1
                    ),
                )
            )

        print(f"Shape of the final features is {_features.shape}")
        self.topological_features = _features

    def save_features(self, path, name):
        if self.topological_features.any() is None:
            raise Exception("Empty topological features, can't save")

        if self.save_spectrograms:
            with open(
                f"{os.path.join(path, name)}_spectrograms.npy", "wb"
            ) as f:
                np.save(f, self.spectrograms)

        with open(f"{os.path.join(path, name)}_topofeatures.npy", "wb") as f:
            np.save(f, self.topological_features)
