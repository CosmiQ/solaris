# Contributing guide for cw-tiler

_This guide is based on [scikit-image's CONTRIBUTING.txt](https://github.com/scikit-image/scikit-image/blob/master/CONTRIBUTING.txt)._

Thank you for contributing to the cw-tiler codebase. The technical details of the development process is summed up below.

---
## Development process
#### 1. If you are a first-time contributor:
   - Go to https://github.com/cosmiq/cw-tiler and click the
     "fork" button to create your own copy of the project.
   - Clone the project to your local computer:
      `git clone https://github.com/your-username/cw-tiler.git`
   - Change the directory:
      `cd cw-tiler`
   - Add the upstream repository::

      git remote add upstream https://github.com/cosmiq/cw-tiler.git
   - Now, you have remote repositories named:
     - `upstream`, which refers to the CosmiQ repository
     - `origin`, which refers to your personal fork

#### 2. Develop your contribution:
   - Pull the latest changes from upstream::
      ```
      git checkout master
      git pull upstream master
      ```
   - Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'add-apls': `git checkout -b add-apls`
   - Commit locally as you progress (you may push to your fork if you wish).

#### 3. To submit your contribution:
   - Push your changes back to your fork on GitHub:
      `git push origin add-apls`
   - Go to your fork of cw-tiler on GitHub. The new branch will show up with a green Pull Request
     button - click it. __When you prepare your pull request, please merge to the `dev` branch in the `upstream` remote.__

#### 4. Review process:

   - Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation and style.  Every single developer working on the project has their code reviewed, and we see it as a conversation from which we all learn and the overall code quality benefits.
   - To update your pull request, make your changes on your local repository and commit. As soon as those changes are pushed up (to the same branch as before) the pull request will update automatically.
   - [Travis-CI](https://travis-ci.org/), a continuous integration service, is triggered after each Pull Request update to build the code, run unit tests, measure code coverage and check coding style (PEP8) of your branch. The Travis tests must pass before your PR can be merged. If Travis fails, you can find out why by clicking on the "failed" icon (red X) and inspecting the build and test log.
   - [Codecov](https://codecov.io/) will run to check how your PR affects test coverage. See the Tests section below for more information.
   - A pull request must be approved by a reviewer before merging.

#### 5. Document changes
   - If your change introduces any API modifications, please update `doc/api_changes.txt`. Note that we don't allow API modifications that _remove existing parameters or alter required arguments_ except in major releases, and if your PR introduces these changes, it will not be merged until the next x.0.0 release.
   - If your change introduces a deprecation, add a reminder to ``TODO.txt``
   for the team to remove the deprecated functionality in the future.

   _Note to reviewers:_ if it is not obvious from the PR description, add a short
   explanation of what a branch did to the merge message and, if closing a
   bug, also add "Closes #123" where 123 is the issue number.


#### Divergence between `upstream master` and your feature branch

If GitHub indicates that the branch of your Pull Request can no longer be merged automatically, merge the master branch into yours:
```
git fetch upstream master
git merge upstream/master
```

If any conflicts occur, they need to be fixed before continuing.  See
which files are in conflict using `git status`, which displayes a message like:
```
Unmerged paths:
  (use "git add <file>..." to mark resolution)

  both modified:   file_with_conflict.txt
```
Inside the conflicted file, you'll find sections like these::
```
<<<<<<< HEAD
The way the text looks in your branch
=======
The way the text looks in the master branch
>>>>>>> master
```
Choose one version of the text that should be kept, and delete the
rest:
```
The way the text looks in your branch
```
Now, add the fixed file with `git add file_with_conflict.txt`, and once you have fixed all of the merge conficts, `git commit`.

_Note:_ Advanced Git users are encouraged to `rebase` instead of `merge`, but we squash most commits either way.

---

## Guidelines
* All code should have tests (see _Test coverage_ below for more details).
* All code should be documented, to the same
  [standard as NumPy and SciPy](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-  standard). See the existing codebase for examples.
- For new functionality, always add an example to the cookbook (`docs/cookbook/`) using a jupyter notebook. Build the docs using [Sphinx](http://www.sphinx-doc.org/en/master/) before you create a PR to make sure your notebook renders properly.
- We encourage you to solicit review from another team member before merging.

#### Stylistic Guidelines

- Follow [PEP08](https://www.python.org/dev/peps/pep-0008/). Check code with pyflakes / flake8.
- Please use the same import conventions as used throughout the rest of the package.

#### Testing
``scikit-image`` has an extensive test suite that ensures correct
execution on your system.  The test suite has to pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.

We make use of the [pytest](https://docs.pytest.org/en/latest/) testing framework, with tests located in the `cw-tiler/tests` folder. You can run the tests yourself (see below) or use Travis-CI test runs in your PR. If you do the latter, begin your PR name with "[WIP]" to mark it as a work in progress until all tests pass.

To use `pytest`, ensure that
the library is installed in development mode using `pip install -e .`

Now, run all tests using `PYTHONPATH=. pytest cw-tiler`

#### Test coverage
Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, install
[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) (using `easy_install pytest-cov`) and then run `make coverage`. This will print a report with one line for each file in `cw_eval`,
detailing the test coverage::
```
Name                                             Stmts   Exec  Cover   Missing
------------------------------------------------------------------------------
skimage/color/colorconv                             77     77   100%
skimage/filter/__init__                              1      1   100%
...
```

#### Activate Travis-CI for your fork (optional)

Travis-CI checks all unittests in the project to prevent breakage.

Before sending a pull request, you may want to check that Travis-CI
successfully passes all tests. To do so,

* Go to [Travis-CI](https://travis-ci.org/) and follow the Sign In link at
  the top
* Go to [your profile page](https://travis-ci.org/profile) and switch on
  your cw-tiler fork

This corresponds to steps one and two in
[Travis-CI documentation](https://about.travis-ci.org/docs/user/getting-started/) (Step three is already done in cw-tiler).

Thus, as soon as you push your code to your fork, it will trigger Travis-CI,
and you will receive an email notification when the process is done. Every time Travis is triggered, it also calls [Codecov](https://codecov.io) to inspect the current test overage.


#### Building docs

To build docs, run `make html` from the `cw-tiler/docs` directory.

Then, all the HTML files will be generated in `cw-tiler/docs/_build/`. To rebuild a full clean documentation, run:
```
make clean
make html
```
__Requirements:__ [Sphinx] (http://www.sphinx-doc.org/en/stable/) is needed to build
the documentation. Sphinx and other python packages needed to build the documentation
can be installed using `pip install .` from within the `cw-tiler` folder.

#### Deprecation cycle

If the behavior of the library has to be changed, a deprecation cycle must be
followed to warn users.

- a deprecation cycle is _not_ necessary when:

    * adding a new function, or
    * adding a new keyword argument to the _end_ of a function signature, or
    * fixing what was buggy behavior

- a deprecation cycle is necessary for _any breaking API change_, meaning a
    change where the function, invoked with the same arguments, would return a
    different result after the change. This includes:

    * changing the order of arguments or keyword arguments, or
    * adding arguments or keyword arguments to a function, or
    * changing a function's name or submodule, or
    * changing the default value of a function's arguments.

Usually, our policy is to put in place a deprecation cycle over two releases.

For the sake of illustration, we consider the modification of a default value in
a function signature. In version N (therefore, next release will be N+1), we
have

```
def a_function(image, rescale=True):
    out = do_something(image, rescale=rescale)
    return out
```
that has to be changed to
```
def a_function(image, rescale=None):
    if rescale is None:
        warn('The default value of rescale will change to `False` in version N+3')
        rescale = True
    out = do_something(image, rescale=rescale)
    return out
```
and in version N+3:
```
def a_function(image, rescale=False):
    out = do_something(image, rescale=rescale)
      return out
```
Here is the process for a 2-release deprecation cycle:

- In the signature, set default to `None`, and modify the docstring to specify
  that it's `True`.
- In the function, _if_ rescale is set to `None`, set to `True` and warn that the
  default will change to `False` in version N+3.
- In ``doc/release/release_dev.rst``, under deprecations, add "In
  `a_function`, the `rescale` argument will default to `False` in N+3."
- In ``TODO.txt``, create an item in the section related to version N+3 and write
  "change rescale default to False in a_function".

Note that the 2-release deprecation cycle is not a strict rule and in some
cases, the developers can agree on a different procedure upon justification
(like when we can't detect the change, or it involves moving or deleting an
entire function for example).

---

## Bugs
Please report bugs [on Github](https://github.com/cw-tiler/issues/)
