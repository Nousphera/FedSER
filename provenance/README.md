# Recovered source

The six files in `original/` are byte-for-byte copies from Vasileios Tsouvalas's
research repository, commit `dbfb6f3da8ada7ef4a63033d77ea10f446cb7091`.
`manifest.json` gives the original paths and SHA-256 checksums. The snapshot
contains the closest recovered implementation of the paper: the federated
semi-supervised baseline. It is an audit reference, not an installable package.
Its old imports, incomplete data dependency and old Flower API are retained
deliberately; use the maintained `fedser` package to run experiments.

All four upstream commits name Vasileios Tsouvalas as author. The source has no
LICENSE file. Publication was authorized by the author; this repository does
not infer an open-source license or third-party redistribution grant from that
authorization. Contact the author for reuse terms until an explicit code
license is added. The paper, IEMOCAP, TensorFlow and Flower retain their own terms.

No private Git history, data, trained weights, logs or result archives are
included. Other explorations (AlexNet, SqueezeNet, LEAF, Light-SERNet and older
archives) are outside this release. The old TensorFlow-derived callback and
Mozilla-derived augmentation helper were not copied or used. The maintained
code uses library APIs for these basic operations and does not need those files.

The maintained model, preprocessing and training code are adapted from the
included source and audited against the paper. Differences and unresolved
choices are listed in [the method mapping](../docs/method.md).
