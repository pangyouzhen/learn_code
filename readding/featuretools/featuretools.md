```sequence
ft.dfs -> dfs
DeepFeatureSynthesis -> dfs_object
dfs_object -> dfs_object._build_entityset
dfs_object._build_features -> self._run_dfs
self._run_dfs -> self.add_identity_features
self._run_dfs -> self.es.get_backward_entities
self._build_features

```
