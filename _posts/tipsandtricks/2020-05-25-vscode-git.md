---
title: Setup vscode and git - Getting rid of typing password
date: 2020-05-25
tags: [github, vscode, setup]
description: Setting vscode with git
categories: tipsandtricks
---

## Why this blog post

If you are annoyed by vscode asking for your password each and every time you commit your changes. You should definitely check out the next set of steps to avoid it.

### Setting up using SSH

#### Step1 - Creating the public and private keys

```bash
cd ~/.ssh/ && ssh-keygen -t rsa -b 4096 -C "replace_with_your_mail_id"
```

#### Step2 - Copying the public ssh key to github

- [adding-a-new-ssh-key-to-your-github-account](https://help.github.com/en/enterprise/2.15/user/articles/adding-a-new-ssh-key-to-your-github-account)

#### Step3 - Cloning your project as a ssh command

In github you will the find the option to clone your repo as a ssh command. Use that format to clone your repository.

```bash
git clone git@github.com:(...Link_to_your_repository...)
```

#### Step4 - Configuring email and username for your cloned repository

```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

## Conclusion

Using the ssh setup is both secure and is a one time setup process. Hope it helps !
