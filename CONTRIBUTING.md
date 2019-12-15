_These contributing guidelines are adapted from [scikit-image](https://github.com/scikit-image/scikit-image) - Copyright 2019, the scikit-image team._

# How to contribute to `Solaris`

We welcome contributions from the open source community! From creating issues to describe bugs or request new features, to PRs to improve the codebase or documentation, we encourage you to dive in, even if you're a novice.

- To find things to work on, check out the [open issues on GitHub](https://github.com/cosmiq/solaris/issues?state=open)
- The technical detail of the development process is summed up below.

## Contributing through issues to identify bugs or request features

We welcome bug reports or feature requests through issues.

1. Go to https://github.com/cosmiq/solaris/issues and search the issues to see if your bug/feature is already present in the list. If not,
2. Create a new issue, using the template appropriate for the type of issue you're creating (bug report/feature request/etc.)
  - Please don't change the labels associated with the issue when you create it - maintainers will do so during triage.
  - If you wish to work on resolving the issue yourself, you're welcome to do so! proceed to the next session for guidelines.

## Contributing through pull requests (PRs) to improve the codebase

1. If you are a first-time contributor:
 - Go to [https://github.com/cosmiq/solaris](https://github.com/cosmiq/solaris) and click the "fork" button to create your own copy of the project.
 - Clone the project to your local computer:
    ```
    git clone https://github.com/your-username/solaris.git
    ```
 - Change the directory:
    ```
    cd solaris
    ```
 - Add the upstream repository:
    ```
    git remote add upstream https://github.com/cosmiq/solaris.git
    ```
 - Now, you have remote repositories named:
   - `upstream`, which refers to the CosmiQ repository
   - `origin`, which refers to your personal fork

2. Develop your contribution:
 - Pull the latest changes from upstream's `dev` branch:
  ```
  git checkout dev
  git pull upstream dev
  ```
 - Create a branch for the issue that you want to work on. (If there isn't already an issue for the bug or feature that you want to implement, create that issue first). We recommend formatting the branch name as `ISS[number]_[short description]`, e.g. `ISS42_meaning`. To do so, run:
  ```
  git checkout -b ISS42_meaning
  ```
 - Commit locally as you progress (``git add`` and ``git commit``)
3. To submit your contribution:
 - Push your changes back to your fork on GitHub:
  ```
  git push origin ISS42_meaning
  ```
 - Enter your GitHub username and password if requested.
 - Go to GitHub. The new branch will show up with a green Pull Request button - click it. Fill out the Pull Request form and click "Submit Pull Request".
 - Monitor the CI tests and debug your code if necessary to ensure that all tests pass.
 - If your PR reduces coverage after tests pass, you may be asked to add new unit tests or extend existing tests. For more, see [Unit Tests](#unit-tests) below.
4. Review process:
 - Core contributors may write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation, and style. This is intended as a friendly conversation from which we all learn and the overall code quality benefits.  Therefore, please don't let the review discourage you from contributing: its only aim is to improve the quality of the project, not to criticize (we are, after all, very grateful for the time you're donating!).
 - To update your pull request, make your changes on your local repository, commit, and push to the same branch on your fork of the repository. As soon as those changes are pushed up (to the same branch as before) the pull request will update automatically.

### Continuing integration
`Travis-CI <https://travis-ci.org/>` (soon to be replaced with GitHub actions), a continuous integration service, is triggered after each Pull Request or new branch push. CI runs unit tests, measures code coverage, and checks coding style (PEP8) of your branch. The Travis tests must pass before your PR can be merged. If Travis fails, you can find out why by clicking on the "failed" icon (red cross) and inspecting the build and test log. The PR will not be merged until the CI run succeeds.

A pull request must be approved by a core team members before merging.

### Unit tests

Our codebase is tested by `pytest` unit tests [in the tests directory](https://github.com/CosmiQ/solaris/tree/master/tests). Those tests run during the pull request CI, and if they fail, the CI fails and the PR will not be merged until it is fixed. When adding new functionality, you are encouraged to extend existing tests or implement new tests to test the functionality you added. As a rule of thumb, any PR should increase code coverage on the repository. If substantial changes are made without accompanying tests, maintainers may ask you to add tests before a PR is merged.

### Document changes

Every pull request must include an update to the "Unreleased" portion of [the changelog](https://github.com/CosmiQ/solaris/blob/master/CHANGELOG.md).

Divergence between ``upstream master`` and your feature branch
--------------------------------------------------------------

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, merge the CosmiQ dev branch into yours:
```
git fetch upstream dev
git merge upstream/dev
```

If any conflicts occur, they need to be fixed before continuing.  See
which files are in conflict using `git status`. This will yield a message like:
```
Unmerged paths:
 (use "git add <file>..." to mark resolution)

 both modified:   file_with_conflict.txt
```

Inside the conflicted file, you'll find sections like these:
```
<<<<<<< HEAD
The way the text looks in your branch
=======
The way the text looks in the master branch
>>>>>>> dev
```
Choose one version of the text that should be kept, and delete the
rest:
```
The way the text looks in your branch
```
Finally, add the fixed files, commit, and push.

## Guidelines

- All code should have tests (see `test coverage`_ below for more details).
- All code should be documented, to the same
  [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) as NumPy and SciPy
- No changes are ever merged into `dev` without review and approval by a maintainer. Maintainers closely monitor pull requests and will usually respond within 24 hours on weekdays, if not faster. __Never merge your own pull request.__

### Stylistic Guidelines

- Follow [PEP008](https://www.python.org/dev/peps/pep-0008/). Check code with pyflakes / flake8.

### Testing

`solaris` has an extensive test suite that ensures correct execution on your system.  The test suite has to pass before a pull request can be merged, and tests should be added to cover any modifications to the code base.

We make use of the [pytest](https://docs.pytest.org/en/latest/)
testing framework, with tests located in the various ``solaris/tests/submodule`` folders. If adding new tests, make sure to add them to the appropriate submodule folder and test script.

### Test coverage

Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%. At a minimum, newly added code should not reduce coverage across the library. To measure the test coverage, install [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) and then run:
```
$ make coverage
```

This will print a report with one line for each file in `solaris`,
detailing the test coverage.

### Activate Travis-CI for your fork (optional)

Travis-CI checks all unit tests in the project to prevent breakage.

Before sending a pull request, you may want to check that Travis-CI
successfully passes all tests. To do so,

- Go to [Travis-CI](https://travis-ci.org/) and follow the Sign In link at
  the top

- Go to your [profile page](https://travis-ci.org/profile) and switch on
  your `solaris` fork

As soon as you push your code to your fork, it will trigger Travis-CI,
and you will receive an email notification when the process is done.

Every time Travis is triggered, it also calls on [Codecov](https://codecov.io) to inspect the current test overage.


### Building docs

Sphinx[http://www.sphinx-doc.org/en/stable/] is needed to build the documentation.
You can install it and other necessary packages with conda.
```
conda install -c conda-forge sphinx sphinx_bootstrap_theme nbsphinx
```

To build docs, run ``make`` from the ``doc`` directory. ``make help`` lists
all targets. For example, to build the HTML documentation, you can run `make html`. Then, all the HTML files will be generated in `solaris/docs/_build/html`. To rebuild a full clean documentation, run:
```
make clean
make html
```

If you have any questions, create an issue.
