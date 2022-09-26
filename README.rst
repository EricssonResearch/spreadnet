
Development Workflow
--------------------

1. If you are a first-time contributor:

   * Go to `https://github.com/theishita/internal-spreadnet/branches
     <https://github.com/theishita/internal-spreadnet/branches>`_, create your
     branch. Since the branch name will appear in the merge message, use a
     sensible name such as 'bugfix-for-issue-1480'.

   * Clone the project to your local computer::

      git clone git@github.com:theishita/internal-spreadnet.git

   * Next, you need to set up your build environment.
     Here are instructions for two popular environment managers:

     * ``venv`` (pip based)

       ::

         # Create a virtualenv named ``spreadnet-dev`` that lives in the directory of
         # the same name
         python -m venv spreadnet-dev
         # Activate it
         source spreadnet-dev/bin/activate
         # Install the dependencies of spreadnet
         pip install -r requirements.txt
         #
         # (Optional) Install pygraphviz and pydot packages
         # These packages require that you have your system properly configured
         # and what that involves differs on various systems.
         # pip install -r requirements/extra.txt
         #
         # Build and install spreadnet from source
         pip install -e .
         # Test your installation
         PYTHONPATH=. pytest spreadnet

     * ``conda`` (Anaconda or Miniconda)

       ::

         # Create a conda environment named ``spreadnet-dev``
         conda create --name spreadnet-dev
         # Activate it
         conda activate spreadnet-dev
         # Install main dependencies
         conda install -c conda-forge --file requirements.txt
         #
         # (Optional) Install pygraphviz and pydot packages
         # These packages require that you have your system properly configured
         # and what that involves differs on various systems.
         # conda install -c conda-forge --file requirements/extra.txt
         #
         # Install spreadnet from source
         pip install -e .
         # Test your installation
         PYTHONPATH=. pytest spreadnet

   * Finally, we recommend you use a pre-commit hook, which runs black and flake8 et al. when
     you type ``git commit``::

       pre-commit install

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git pull

   * Navigate to the folder ``spreadnet`` and switch to the branch you created.

   * Commit locally as you progress (``git add`` and ``git commit``)

3. If introducing a new feature or patching a bug, be sure to add new test cases
   in the relevant file in ``spreadnet/tests/``.

4. Test your contribution:

   * Run the test suite locally (see `Testing`_ for details)::

      PYTHONPATH=. pytest spreadnet

   * Running the tests locally *before* submitting a pull request helps catch
     problems early and reduces the load on the continuous integration
     system.

5. Submit your contribution:

   * Push your changes back to the repo on GitHub::

      git push

   * Go to GitHub. The new branch will show up with a green Pull Request
     button---click it.


6. Review process:

   * When you first create a PR, add at least two reviewer to the `reviewer` section.

   * Reviewers will write inline and/or general comments on your PR to help
     you improve its implementation, documentation, and style. Reviewer should
     add the ``@author-action-required`` label if further actions are required.

   * To update your PR, make your changes on your local repository
     and commit, and remove the ``@author-action-required`` label from the PR.
     As soon as those changes are pushed up (to the same branch as before) the
     PR will update automatically.

   * Repeat this process until assignees approve your PR.

   * Once the PR is approved, the author is in charge of ensuring the PR passes
     the build. Add the ``test-ok`` label if the build succeeds.

   * Committers will merge the PR once the build is passing.

   * Every Pull Request (PR) update triggers a set of `continuous integration
     <https://en.wikipedia.org/wiki/Continuous_integration>`_ services
     that check that the code is up to standards and passes all our tests.
     These checks must pass before your PR can be merged.  If one of the
     checks fails, you can find out why by clicking on the "failed" icon (red
     cross) and inspecting the build and test log.

   .. note::

      If the PR closes an issue, make sure that GitHub knows to automatically
      close the issue when the PR is merged.  For example, if the PR closes
      issue number 1480, you could use the phrase "Fixes #1480" in the PR
      description or commit message.


Divergence from ``main``
---------------------------------

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, merge the main branch into yours::

   git merge main

If any conflicts occur, they need to be fixed before continuing.  See
which files are in conflict using::

   git status

Which displays a message like::

   Unmerged paths:
     (use "git add <file>..." to mark resolution)

     both modified:   file_with_conflict.txt

Inside the conflicted file, you'll find sections like these::

   <<<<<<< HEAD
   The way the text looks in your branch
   =======
   The way the text looks in the main branch
   >>>>>>> main

Choose one version of the text that should be kept, and delete the
rest::

   The way the text looks in your branch

Now, add the fixed file::


   git add file_with_conflict.txt

Once you've fixed all merge conflicts, do::

   git commit

.. note::

   Advanced Git users may want to rebase instead of merge,
   but we squash and merge PRs either way.


Guidelines
----------

* Don't forget to install pre-commit hooks on the root folder if you haven't done so:

       pre-commit install

* Except from neural networks training, all code should have tests.
* All code should follow the same
  `standards <https://google.github.io/styleguide/pyguide.html>`__
  as Google style guide. For Python documentation, we follow a subset of the
  `Google pydoc format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__.


* All changes are reviewed.  Ask on `Slack` if
  you get no response to your pull request.

..   TODO
.. * Default dependencies are listed in ``requirements/default.txt`` and extra
..   (i.e., optional) dependencies are listed in ``requirements/extra.txt``.
..   We don't often add new default and extra dependencies.  If you are considering
..   adding code that has a dependency, you should first consider adding a gallery
..   example.  Typically, new proposed dependencies would first be added as extra
..   dependencies.  Extra dependencies should be easy to install on all platforms
..   and widely-used.

Testing
-------

``spreadnet`` uses a test suite that ensures correct
execution on your system.  The test suite has to pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.
We make use of the `pytest <https://docs.pytest.org/en/latest/>`__
testing framework.

To run all tests::

    $ PYTHONPATH=. pytest spreadnet

.. TODO: coverage test
.. TODO: CI test
.. TODO: doctest


(TODO) Adding examples
-------------------------

The gallery examples are managed by
`sphinx-gallery <https://sphinx-gallery.readthedocs.io/>`_.
The source files for the example gallery are ``.py`` scripts in ``examples/`` that
generate one or more figures. They are executed automatically by sphinx-gallery when the
documentation is built. The output is gathered and assembled into the gallery.

You can **add a new** plot by placing a new ``.py`` file in one of the directories inside the
``examples`` directory of the repository. See the other examples to get an idea for the
format.

.. note:: Gallery examples should start with ``plot_``, e.g. ``plot_new_example.py``

General guidelines for making a good gallery plot:

* Examples should highlight a single feature/command.
* Try to make the example as simple as possible.
* Data needed by examples should be included in the same directory and the example script.
* Add comments to explain things are aren't obvious from reading the code.
* Describe the feature that you're showcasing and link to other relevant parts of the
  documentation.



Bugs
----

Please `report bugs on GitHub <https://github.com/theishita/internal-spreadnet>`_.
