initSidebarItems({"mod":[["exponential","The exponential distribution."],["gamma","The Gamma and derived distributions."],["normal","The normal and derived distributions."],["range","Generating numbers between two others."]],"struct":[["RandSample","A wrapper for generating types that implement `Rand` via the `Sample` & `IndependentSample` traits."],["Weighted","A value with a particular weight for use with `WeightedChoice`."],["WeightedChoice","A distribution that selects from a finite collection of weighted items.Each item has an associated weight that influences how likely it is to be chosen: higher weight is more likely.The `Clone` restriction is a limitation of the `Sample` and `IndependentSample` traits. Note that `&T` is (cheaply) `Clone` for all `T`, as is `u32`, so one can store references or indices into another vector.Example"]],"trait":[["IndependentSample","`Sample`s that do not require keeping track of state.Since no state is recorded, each sample is (statistically) independent of all others, assuming the `Rng` used has this property."],["Sample","Types that can be used to create a random instance of `Support`."]]});