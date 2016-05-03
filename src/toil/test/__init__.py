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
import logging
import os
import tempfile
import unittest
import shutil
import re
import subprocess

from bd2k.util.files import mkdir_p

from toil import toilPackageDirPath

log = logging.getLogger(__name__)


class ToilTest(unittest.TestCase):
    """
    A common base class for Toil tests. Please have every test case directly or indirectly
    inherit this one.

    When running tests you may optionally set the TOIL_TEST_TEMP environment variable to the path
    of a directory where you want temporary test files be placed. The directory will be created
    if it doesn't exist. The path may be relative in which case it will be assumed to be relative
    to the project root. If TOIL_TEST_TEMP is not defined, temporary files and directories will
    be created in the system's default location for such files and any temporary files or
    directories left over from tests will be removed automatically removed during tear down.
    Otherwise, left-over files will not be removed.
    """

    _tempBaseDir = None

    @classmethod
    def setUpClass(cls):
        super(ToilTest, cls).setUpClass()
        cls._tempDirs = []
        tempBaseDir = os.environ.get('TOIL_TEST_TEMP', None)
        if tempBaseDir is not None and not os.path.isabs(tempBaseDir):
            tempBaseDir = os.path.abspath(os.path.join(cls._projectRootPath(), tempBaseDir))
            mkdir_p(tempBaseDir)
        cls._tempBaseDir = tempBaseDir

    @classmethod
    def _getUtilScriptPath(cls, script_name):
        return os.path.join(toilPackageDirPath(), 'utils', script_name + '.py')

    @classmethod
    def _projectRootPath(cls):
        """
        Returns the path to the project root, i.e. the directory that typically contains the .git
        and src subdirectories. This method has limited utility. It only works if in "develop"
        mode, since it assumes the existence of a src subdirectory which, in a regular install
        wouldn't exist. Then again, in that mode project root has no meaning anyways.
        """
        assert re.search(r'__init__\.pyc?$', __file__)
        projectRootPath = os.path.dirname(os.path.abspath(__file__))
        packageComponents = __name__.split('.')
        expectedSuffix = os.path.join('src', *packageComponents)
        assert projectRootPath.endswith(expectedSuffix)
        projectRootPath = projectRootPath[:-len(expectedSuffix)]
        return projectRootPath

    @classmethod
    def tearDownClass(cls):
        if cls._tempBaseDir is None:
            while cls._tempDirs:
                tempDir = cls._tempDirs.pop()
                if os.path.exists(tempDir):
                    shutil.rmtree(tempDir)
        else:
            cls._tempDirs = []
        super(ToilTest, cls).tearDownClass()

    def setUp(self):
        log.info("Setting up %s ...", self.id())
        super(ToilTest, self).setUp()

    def _createTempDir(self, purpose=None):
        prefix = ['toil', 'test', self.id()]
        if purpose: prefix.append(purpose)
        prefix.append('')
        temp_dir_path = tempfile.mkdtemp(dir=self._tempBaseDir, prefix='-'.join(prefix))
        self._tempDirs.append(temp_dir_path)
        return temp_dir_path

    def tearDown(self):
        super(ToilTest, self).tearDown()
        log.info("Tore down %s", self.id())

    def _getTestJobStorePath(self):
        path = self._createTempDir(purpose='jobstore')
        # We only need a unique path, directory shouldn't actually exist. This of course is racy
        # and insecure because another thread could now allocate the same path as a temporary
        # directory. However, the built-in tempfile module randomizes the name temp dir suffixes
        # reasonably well (1 in 63 ^ 6 chance of collision), making this an unlikely scenario.
        os.rmdir(path)
        return path


try:
    # noinspection PyUnresolvedReferences
    from _pytest.mark import MarkDecorator
except ImportError:
    # noinspection PyUnusedLocal
    def _mark_test(name, test_item):
        return test_item
else:
    def _mark_test(name, test_item):
        return MarkDecorator(name)(test_item)


def needs_aws(test_item):
    """
    Use as a decorator before test classes or methods to only run them if AWS usable.
    """
    test_item = _mark_test('aws', test_item)
    try:
        # noinspection PyUnresolvedReferences
        import boto
    except ImportError:
        return unittest.skip("Skipping test. Install toil with the 'aws' extra to include this "
                             "test.")(test_item)
    except:
        raise
    else:
        dot_boto_path = os.path.expanduser('~/.boto')
        dot_aws_credentials_path = os.path.expanduser('~/.aws/credentials')
        hv_uuid_path = '/sys/hypervisor/uuid'
        if (os.path.exists(dot_boto_path)
            or os.path.exists(dot_aws_credentials_path)
            # Assume that EC2 machines like the Jenkins slave that we run CI on will have IAM roles
            or os.path.exists(hv_uuid_path) and file_begins_with(hv_uuid_path,'ec2')):
            return test_item
        else:
            return unittest.skip("Skipping test. Create ~/.boto or ~/.aws/credentials to include "
                                 "this test.")(test_item)


def file_begins_with(path, prefix):
    with open(path) as f:
        return f.read(len(prefix)) == prefix


def needs_azure(test_item):
    """
    Use as a decorator before test classes or methods to only run them if Azure is usable.
    """
    test_item = _mark_test('azure', test_item)
    try:
        # noinspection PyUnresolvedReferences
        import azure.storage
    except ImportError:
        return unittest.skip("Skipping test. Install toil with the 'azure' extra) to include this "
                             "test.")(test_item)
    except:
        raise
    else:
        from toil.jobStores.azureJobStore import credential_file_path
        full_credential_file_path = os.path.expanduser(credential_file_path)
        if not os.path.exists(full_credential_file_path):
            return unittest.skip("Skipping test. Configure %s with the access key for the "
                                 "'toiltest' storage account." % credential_file_path)(test_item)
        return test_item


def needs_gridengine(test_item):
    """
    Use as a decorator before test classes or methods to only run them if GridEngine is installed.
    """
    test_item = _mark_test('gridengine', test_item)
    try:
        with open(os.devnull, 'r+') as devnull:
            subprocess.Popen('qsub', stdout=devnull, stderr=devnull, stdin=devnull)
    except OSError:
        return unittest.skip("Skipping test. Install GridEngine to include this test.")(test_item)
    except:
        raise
    else:
        return test_item


def needs_mesos(test_item):
    """
    Use as a decorator before test classes or methods to only run them if the Mesos is installed
    and configured.
    """
    test_item = _mark_test('mesos', test_item)
    try:
        # noinspection PyUnresolvedReferences
        import mesos.native
    except ImportError:
        return unittest.skip("Skipping test. Install toil with the 'mesos' extra to include this "
                             "test.")(test_item)
    except:
        raise
    else:
        return test_item


def needs_parasol(test_item):
    """
    Use as decorator so tests are only run if Parasol is installed.
    """
    test_item = _mark_test('parasol', test_item)
    try:
        with open(os.devnull, 'r+') as devnull:
            subprocess.Popen('parasol', stdout=devnull, stderr=devnull, stdin=devnull)
    except OSError:
        return unittest.skip("Skipping test. Install Parasol to include this test.")(test_item)
    except:
        raise
    else:
        return test_item


def needs_slurm(test_item):
    """
    Use as a decorator before test classes or methods to only run them if Slurm is installed.
    """
    test_item = _mark_test('slurm', test_item)
    try:
        with open(os.devnull, 'r+') as devnull:
            subprocess.Popen('squeue', stdout=devnull, stderr=devnull, stdin=devnull)
    except OSError:
        return unittest.skip("Skipping test. Install Slurm to include this test.")(test_item)
    except:
        raise
    else:
        return test_item


def needs_encryption(test_item):
    """
    Use as a decorator before test classes or methods to only run them if PyNaCl is installed
    and configured.
    """
    test_item = _mark_test('encryption', test_item)
    try:
        # noinspection PyUnresolvedReferences
        import nacl
    except ImportError:
        return unittest.skip("Skipping test. Install toil with the 'encryption' extra to include "
                             "this test.")(test_item)
    except:
        raise
    else:
        return test_item


def needs_cwl(test_item):
    """
    Use as a decorator before test classes or methods to only run them if CWLTool is installed
    and configured.
    """
    test_item = _mark_test('cwl', test_item)
    try:
        # noinspection PyUnresolvedReferences
        import cwltool
    except ImportError:
        return unittest.skip("Skipping test. Install toil with the 'cwl' extra to include this "
                             "test.")(test_item)
    except:
        raise
    else:
        return test_item

methodNamePartRegex = re.compile('^[a-zA-Z_0-9]+$')
# FIXME: move to bd2k-python-lib


def make_tests(generalMethod, targetClass=None, **kwargs):
    """
    This method dynamically generates test methods using the generalMethod as a template. Each generated
    function is the result of a unique combination of parameters applied to the generalMethod. Each of the
    parameters has a corresponding string that will be used to name the method. These generated functions
    are named in the scheme:
        test_[generalMethodName]___[firstParamaterName]_[someValueName]__[secondParamaterName]_...

    The arguments following the generalMethodName should be a series of one or more dictionaries of the form
    {str : type, ...} where the key represents the name of the value. The names will be used to represent the
    permutation of values passed for each parameter in the generalMethod.

    :param generalMethod: A method that will be parametrized with values passed as kwargs. Note that the
        generalMethod must be a regular method.

    :param targetClass: This represents the class to which the generated test methods will be bound. If no
        targetClass is specified the class of the generalMethod is assumed the target.

    :param kwargs: a series of dictionaries defining values, and their respective names where each keyword is
        the name of a parameter in generalMethod.

    >>> class Foo:
    ...     def has(self, num, letter):
    ...         return num, letter
    ...
    ...     def hasOne(self, num):
    ...         return num

    >>> class Bar(Foo):
    ...     pass

    >>> make_tests(Foo.has, targetClass=Bar, num={'one':1, 'two':2}, letter={'a':'a', 'b':'b'})

    >>> b = Bar()

    >>> assert b.test_has__num_One__letter_A() == b.has(1, 'a')

    >>> assert b.test_has__num_One__letter_B() == b.has(1, 'b')

    >>> assert b.test_has__num_Two__letter_A() == b.has(2, 'a')

    >>> assert b.test_has__num_Two__letter_B() == b.has(2, 'b')

    >>> f = Foo()

    >>> hasattr(f, 'test_has__num_One__letter_A')  # should be false because Foo has no test methods
    False

    >>> make_tests(Foo.has, num={'one':1, 'two':2}, letter={'a':'a', 'b':'b'})

    >>> hasattr(f, 'test_has__num_One__letter_A')
    True

    >>> assert f.test_has__num_One__letter_A() == f.has(1, 'a')

    >>> assert f.test_has__num_One__letter_B() == f.has(1, 'b')

    >>> assert f.test_has__num_Two__letter_A() == f.has(2, 'a')

    >>> assert f.test_has__num_Two__letter_B() == f.has(2, 'b')

    >>> make_tests(Foo.hasOne, num={'one':1, 'two':2})

    >>> assert f.test_hasOne__num_One() == f.hasOne(1)

    >>> assert f.test_hasOne__num_Two() == f.hasOne(2)

    """
    def pop(d):
        """
        Pops an arbitrary key value pair from the dict
        :param d: a dictionary
        :return: the popped key, value tuple
        """
        k, v = next(kwargs.iteritems())
        del d[k]
        return k, v

    def permuteIntoLeft(left, rPrmName, right):
        """
        Permutes values in right dictionary into each parameter: value dict pair in the left dictionary.
        Such that the left dictionary will contain a new set of keys each of which is a combination of one of
        its original parameter-value names appended with some parameter-value name from the right dictionary.
        Each original key in the left is deleted from the left dictionary after the permutation of the key and
        every parameter-value name from the right has been added to the left dictionary.

        For example
        if left is {'__PrmOne_ValName':{'ValName':Val}} and right is {'rValName1':rVal1, 'rValName2':rVal2} then
        left will become
        {'__PrmOne_ValName__rParamName_rValName1':{'ValName':Val. 'rValName1':rVal1},
        '__PrmOne_ValName__rParamName_rValName2':{'ValName':Val. 'rValName2':rVal2}}

        :param left: A dictionary pairing each paramNameValue to a nested dictionary that contains each ValueName
            and value pair described in the outer dict's paramNameValue key.
        :param rParamName: The name of the parameter that each value in the right dict represents.
        :param right: A dict that pairs 1 or more valueNames and values for the rParamName parameter.
        """
        for prmValName, lDict in left.items():
            for rValName, rVal in right.items():
                nextPrmVal = ('__%s_%s' % (rPrmName, rValName.title()))
                if methodNamePartRegex.match(nextPrmVal) is None:
                    raise RuntimeError("The name '%s' cannot be used in a method name" % pvName)
                aggDict = dict(lDict)
                aggDict[rPrmName] = rVal
                left[prmValName + nextPrmVal] = aggDict
            left.pop(prmValName)

    def insertMethodToClass():
        """
        Generates and inserts test methods.
        """
        def fx(self, prms=prms):
            if prms is not None:
                return generalMethod(self, **prms)
            else:
                return generalMethod(self)
        setattr(targetClass, 'test_%s%s' % (generalMethod.__name__, prmNames), fx)

    if len(kwargs) > 0:
        # create first left dict
        left = {}
        prmName, vals = pop(kwargs)
        for valName, val in vals.items():
            pvName = '__%s_%s' % (prmName, valName.title())
            if methodNamePartRegex.match(pvName) is None:
                raise RuntimeError("The name '%s' cannot be used in a method name" % pvName)
            left[pvName] = {prmName: val}

        # get cartesian product
        while len(kwargs) > 0:
            permuteIntoLeft(left, *pop(kwargs))

        # set class attributes
        targetClass = targetClass or generalMethod.im_class
        for prmNames, prms in left.items():
            insertMethodToClass()
    else:
        prms = None
        prmNames = ""
        insertMethodToClass()
