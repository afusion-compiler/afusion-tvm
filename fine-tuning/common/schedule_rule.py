# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""A lightweight wrapper on an arbitrary function that can be used to schedule a TIR PrimFunc."""
from typing import Callable, List, Union

from tvm import tir
from tvm.target import Target


class ScheduleRule:  

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        tunable: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        raise NotImplementedError

    @staticmethod
    def from_callable(
        name,
    ) -> Callable[
        [
            Callable[
                [tir.PrimFunc, Target, bool],
                Union[None, tir.Schedule, List[tir.Schedule]],
            ],
        ],
        "ScheduleRule",
    ]:

        def decorator(f) -> "ScheduleRule":  # pylint: disable=invalid-name
            class _Rule(ScheduleRule):
                def apply(
                    self,
                    func: tir.PrimFunc,
                    target: Target,
                    tunable: bool,
                ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
                    return f(func, target, tunable)

            _Rule.__name__ = name
            return _Rule()

        return decorator
