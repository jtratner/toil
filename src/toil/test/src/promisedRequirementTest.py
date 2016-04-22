# Copyright (C) 2015 UCSC Computational Genomics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import os
import fcntl
import time
import logging
import multiprocessing
from toil.common import Config
from toil.batchSystems.singleMachine import SingleMachineBatchSystem
from toil.batchSystems.mesos.test import MesosTestSupport
from toil.job import Job
from toil.job import PromisedRequirement
from toil.test import ToilTest, needs_mesos

log = logging.getLogger(__name__)


class hidden:
    """
    Hide abstract base class from unittest's test case loader

    http://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class#answer-25695512
    """

    class AbstractPromisedRequirementsTest(ToilTest):
        """
        A base test case with generic tests that every batch system should pass

        Uses PromisedRequirement object to allocate core resource requirements.
        """

        __metaclass__ = ABCMeta

        cpu_count = multiprocessing.cpu_count()

        @abstractmethod
        def createBatchSystem(self):
            """
            :rtype: (str, AbstractBatchSystem)
            """
            raise NotImplementedError

        def _createDummyConfig(self):
            return Config()

        def setUp(self):
            self.config = self._createDummyConfig()
            self.batchSystemName, self.batchSystem = self.createBatchSystem()
            super(hidden.AbstractPromisedRequirementsTest, self).setUp()

        def tearDown(self):
            self.batchSystem.shutdown()
            super(hidden.AbstractPromisedRequirementsTest, self).tearDown()

        def testPromisedRequirementDynamic(self):
            for allocated_cores in [1, 2]:
                temp_dir = self._createTempDir('testFiles')

                options = Job.Runner.getDefaultOptions(self._getTestJobStorePath())
                options.logLevel = "DEBUG"
                options.batchSystem = self.batchSystemName
                options.workDir = self._createTempDir('testFiles')

                counter_path = os.path.join(temp_dir, 'counter')
                resetCounters(counter_path)
                min_value, max_value = getCounters(counter_path)
                assert (min_value, max_value) == (0, 0)
                root = Job.wrapJobFn(max_concurrency, self.cpu_count, counter_path, allocated_cores,
                                     cores=self.cpu_count, memory='1G', disk='1G')
                value = Job.Runner.startToil(root, options)
                self.assertEqual(value, self.cpu_count / allocated_cores)

        def testPromisedRequirementStatic(self):
            for allocated_cores in [1, 2]:
                temp_dir = self._createTempDir('testFiles')

                options = Job.Runner.getDefaultOptions(self._getTestJobStorePath())
                options.logLevel = "DEBUG"
                options.batchSystem = self.batchSystemName
                options.workDir = temp_dir

                counter_path = os.path.join(temp_dir, 'counter')
                resetCounters(counter_path)
                min_value, max_value = getCounters(counter_path)
                assert (min_value, max_value) == (0, 0)

                root = Job()
                one1 = Job.wrapFn(one, cores=0.1, memory='1M', disk=1001)
                one2 = Job.wrapFn(one, cores=0.1, memory='1M', disk=1001)
                mb1 = Job.wrapFn(oneMB, cores=0.1, memory='1M', disk=1001)
                root.addChild(one1)
                root.addChild(one2)
                root.addChild(mb1)
                for _ in range(self.cpu_count):
                    root.addFollowOn(Job.wrapFn(measure_concurrency, counter_path,
                                                cores=PromisedRequirement(lambda x: x * allocated_cores, one1.rv()),
                                                memory=PromisedRequirement(mb1.rv()),
                                                disk=PromisedRequirement(lambda x, y: x + y + 1022, one1.rv(), one2.rv())))
                Job.Runner.startToil(root, options)
                min_value, max_value = getCounters(counter_path)
                self.assertEqual(max_value, self.cpu_count / allocated_cores)


def max_concurrency(job, cpu_count, filename, allocated_cores):
    """
    Returns the max number of concurrent tasks when using a PromisedRequirement instance
    to allocate the number of cores per job.

    :param int cpu_count: number of available cpus
    :param str filename: path to counter file
    :param int allocated_cores: number of cores assigned to each job
    :return int max concurrency value:
    """
    one1 = job.addChildFn(one, cores=0.1, memory='32M', disk=1001)
    one2 = job.addChildFn(one, cores=0.1, memory='32M', disk=1001)
    mb1 = job.addChildFn(oneMB, cores=0.1, memory='32M', disk=1001)

    values = []
    for _ in range(cpu_count):
        value = job.addFollowOnFn(measure_concurrency, filename,
                                  cores=PromisedRequirement(lambda x: x * allocated_cores, one1.rv()),
                                  memory=PromisedRequirement(mb1.rv()),
                                  disk=PromisedRequirement(lambda x, y: x + y + 1022, one1.rv(), one2.rv())).rv()
        values.append(value)
    return max(values)


def one():
    return 1

def oneMB():
    return '1M'


def measure_concurrency(filepath):
    """
    Run in parallel to test the number of concurrent tasks.
    This code was copied from toil.batchSystemTestMaxCoresSingleMachineBatchSystemTest
    :param str filepath: path to counter file
    :return int max concurrency value:
    """
    count(1, filepath)
    try:
        time.sleep(5)
    finally:
        return count(-1, filepath)


def count(delta, file_path="counter"):
    """
    Increments data in counter file and tracks the max number of concurrent tasks.
    Counter data must be in the form concurrent tasks, max concurrent tasks
    (counter should be initialized to 0,0)

    :param int delta: increment value
    :param str file_path: path to shared counter file
    :return int max concurrent tasks:
    """
    fd = os.open(file_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            s = os.read(fd, 10)
            value, maxValue = map(int, s.split(','))
            value += delta
            if value > maxValue: maxValue = value
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, ','.join(map(str, (value, maxValue))))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return maxValue


def getCounters(path):
    with open(path, 'r+') as f:
        s = f.read()
        concurrentTasks, maxConcurrentTasks = map(int, s.split(','))
    return concurrentTasks, maxConcurrentTasks


def resetCounters(path):
    with open(path, "w") as f:
        f.write("0,0")
        f.close()


class SingleMachinePromisedRequirementsTest(hidden.AbstractPromisedRequirementsTest):
    """
    Tests against the SingleMachine batch system
    """

    def createBatchSystem(self):
        return "singleMachine", SingleMachineBatchSystem(config=self.config,
                                                         maxCores=self.cpu_count,
                                                         maxMemory=1e9, maxDisk=1001)


@needs_mesos
class MesosPromisedRequirementsTest(hidden.AbstractPromisedRequirementsTest, MesosTestSupport):
    """
    Tests against the Mesos batch system
    """

    def createBatchSystem(self):
        from toil.batchSystems.mesos.batchSystem import MesosBatchSystem
        self._startMesos()
        return "mesos", MesosBatchSystem(config=self.config,
                                         maxCores=self.cpu_count, maxMemory=1e9, maxDisk=1001,
                                         masterAddress='127.0.0.1:5050')

    def tearDown(self):
        self._stopMesos()
        super(MesosPromisedRequirementsTest, self).tearDown()



