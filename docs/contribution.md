Contribution guide
=============

pyJedAI is a git repository hosted on GitHub. If you want to contribute to the project, you can follow the steps below.

1. __Fork the repository__: click on the `Fork` button on the top right corner of the repository page. This will create a copy of the repository in your GitHub account.
2. __Clone the repository__: `git clone ` followed by the URL of your forked repository.
3. __Create a new branch__: `git checkout -b <branch_name>` where `<branch_name>` is the name of the branch you want to create. The name of the branch should be related to the feature you want to implement or the bug you want to fix. For example, if you want to implement a new feature, you can name the branch `feature/<feature_name>`.
4. __Make changes__: add the files you want to commit to the staging area with `git add <file_name>` and commit them with `git commit -m "<commit_message>"`. The commit message should be short and descriptive. If you want to add a new file, you can use `git add .` to add all the files in the current directory.
5. __Commit and push__: `git push origin <branch_name>` to push the changes to your forked repository. If you want to add more commits, you can repeat steps 4 and 5. 
6. __Create a pull request__: go to the repository page on GitHub and click on the `New pull request` button. Select the branch you want to merge into the main repository and click on `Create pull request`.
7. __Wait for the review__: the pull request will be reviewed by the maintainers of the repository. If the pull request is approved, it will be merged into the main repository. Otherwise, you will be asked to make some changes.
8. __Merge__: once the pull request is approved, you can merge it into the main repository by clicking on the `Merge pull request` button.
9. __Sync your fork__: if you want to keep your forked repository up to date with the main repository, you can follow the steps below.
    1. Add the main repository as a remote: `git remote add upstream ` followed by the URL of the main repository.
    2. Fetch the changes: `git fetch upstream`.
    3. Merge the changes: `git merge upstream/main`.
    4. Push the changes to your forked repository: `git push origin main`.


