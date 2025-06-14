# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from ...datasource import ones


def test_reshape():
    a = ones((10, 600), chunk_size=5)
    a.shape = [10, 30, 20]

    # test reshape unknown shape
    c = a[a > 0]
    d = c.reshape(10, 600)
    assert d.shape == (10, 600)
    d = c.reshape(-1, 10)
    assert len(d.shape) == 2
    assert np.isnan(d.shape[0])
    assert d.shape[1]

    with pytest.raises(TypeError):
        a.reshape((10, 30, 20), other_argument=True)
