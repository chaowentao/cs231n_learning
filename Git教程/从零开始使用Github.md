本文是学习完廖雪峰老师[Git教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)后的精简实用版总结，再次对廖雪峰老师表示感谢。
## 1 安装git
Git是目前最简单的版本管理工具，并且可以在Linux、Unix、Mac和Windows这几大平台上正常运行了。要使用Git，第一步当然是安装Git了。

**在Linux(Ubuntu)上安装Git**
```
$ sudo apt-get install git
```
一条指令直接完成Git的安装，非常简单。

**在Mac OS上安装Git**

直接从AppStore安装Xcode，Xcode集成了Git，不过默认没有安装，你需要运行Xcode，选择菜单“Xcode”->“Preferences”，在弹出窗口中找到“Downloads”，选择“Command Line Tools”，点“Install”就可以完成安装了。

**在Windows上安装Git**

从Git官网直接下载[安装程序](https://git-scm.com/downloads)，（网速慢的请移步[国内镜像](https://pan.baidu.com/s/1kU5OCOB#list/path=%2Fpub%2Fgit)），然后按默认选项安装即可。安装完成后，在开始菜单里找到“Git”->“Git Bash”，蹦出一个类似命令行窗口的东西，就说明Git安装成功！

安装完成后，还需要最后一步设置，在命令行输入
```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```
注意``git config``命令的``--global``参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址。

## 2 创建版本库
版本库可以理解为一个目录，所有文件都可以被Git管理起来
> 如果使用Windows系统，推荐在git bash下进行操作。

第一步，创建一个空目录：
```
$ mkdir learngit
$ cd learngit
$ pwd
/e/learngit
```
``pwd``命令用于显示当前目录。在我的Windows上，这个仓库位于``/e/learngit``。

> 如果使用Windows系统，为了避免遇到各种问题，请确保目录名（包括父目录）不包含中文。

第二步，通过``git init``命令把这个目录变成Git可以管理的仓库：
```
$ git init
Initialized empty Git repository in E:/learngit/.git/
```
第三步，添加文件到版本库

在``learngit``目录下创建``readme.txt``文件，内容如下：
```
Git is a version control system.
Git is free software.
```
> 如果使用Windows系统，不要使用记事本编辑，推荐使用Notepad++。

然后，用命令``git add``告诉Git，把文件添加到仓库：
```
$ git add readme.txt
```
执行上面的命令，没有任何显示，说明添加成功。

最后，用命令``git commit``告诉Git，把文件提交到仓库：
```
$ git commit -m "wrote a readme file"
[master (root-commit) 50b9e02] wrote a readme file
 1 file changed, 2 insertions(+)
 create mode 100644 readme.txt
```
``git commit``命令中``-m``后面输入的是本次提交的说明，方便你在历史记录中找到改动记录。

``git commit``命令执行成功后会告诉你，``1 file changed``：1个文件被改动（我们新添加的readme.txt文件）；``2 insertions``：插入了两行内容（readme.txt有两行内容）

为什么Git添加文件需要``add``，``commit``一共两步呢？因``为commit``可以一次提交很多文件，所以你可以多次``add``不同的文件

使用``git status``命令，可以随时掌握工作区的状态

如果``git status``告诉你有文件被修改过，用``git diff``可以查看修改内容。

## 3 时光穿梭机
### 版本回退

使用``git log``可以查看提交历史

在Git中，用``HEAD``表示当前版本，也就是最新的提交``1094adb...``（注意我的提交ID和你的肯定不一样），上一个版本就是``HEAD^``，上上一个版本就是``HEAD^^``，当然往上100个版本写100个``^``比较容易数不过来，所以写成``HEAD~100``。

使用``git reset``命令进行``HEAD``头指针跳转：
```
$ git reset --hard HEAD^
HEAD is now at 3aeb22f append GPL
```

使用``git reset``命令进行版本号跳转：
```
$ git reset --hard 3aeb22
HEAD is now at 3aeb22f append GPL
```
使用``git reflog``查看命令历史，以便确定要回到哪个版本

### 撤销修改

场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令``git checkout -- file``。
```
$ git checkout -- readme.txt
```
场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令``git reset HEAD <file>``，就回到了场景1，第二步按场景1操作。
```
$ git reset HEAD readme.txt
Unstaged changes after reset:
M    readme.txt
```
场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考版本回退一节，不过前提是没有推送到远程库。。

命令``git rm``用于删除一个文件。如果一个文件已经被提交到版本库，那么你永远不用担心误删，但是要小心，你只能恢复文件到最新版本，你会丢失最近一次提交后你修改的内容。
```
$ git rm test.txt
rm 'test.txt'

$ git commit -m "remove test.txt"
[master d46f35e] remove test.txt
 1 file changed, 1 deletion(-)
 delete mode 100644 test.txt
```
## 4 远程仓库
自行注册GitHub账号。由于本地Git仓库和GitHub仓库之间的传输是通过SSH加密的，所以需要设置：

第1步：创建SSH Key。在用户主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果已经有了，可直接跳到下一步。如果没有，打开Shell（Windows下打开Git Bash），创建SSH Key：
```
$ ssh-keygen -t rsa -C "youremail@example.com"
```
如果一切顺利的话，可以在用户主目录里找到.ssh目录，里面有``id_rsa``和``id_rsa.pub``两个文件，这两个就是SSH Key的秘钥对，``id_rsa``是私钥，不能泄露出去，``id_rsa.pub``是公钥，可以放心地告诉任何人。

第2步：登陆GitHub，打开“Account settings”，“SSH Keys”页面，然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴``id_rsa.pub``文件的内容。

GitHub允许你添加多个Key。假定你有若干电脑，你一会儿在公司提交，一会儿在家里提交，只要把每台电脑的Key都添加到GitHub，就可以在每台电脑上往GitHub推送了。

### 添加远程库 ##
在本地创建了一个Git仓库后，又想在GitHub创建一个Git仓库，并且让这两个仓库进行远程同步，这样，GitHub上的仓库既可以作为备份，又可以让其他人通过该仓库来协作。

首先，登陆GitHub，然后，在右上角找到“Create a new repo”按钮，创建一个新的仓库；
在Repository name填入``learngit``，其他保持默认设置，点击“Create repository”按钮，就成功地创建了一个新的Git仓库;

要关联一个远程库，使用命令``git remote``进行关联
```
git remote add origin git@server-name:path/repo-name.git；
```
关联后，使用命令``git push``推送master分支的所有内容；

第一次推送
```
git push -u origin master
```
以后修改推送
```
git push origin master
```
### 克隆远程库 ##
要克隆一个仓库，首先必须知道仓库的地址，然后使用``git clone``命令克隆。

Git支持多种协议，包括``https``，但通过``ssh``支持的原生``git``协议速度最快。
```
$ git clone git@github.com:chaowentao/gitskillls.git
Cloning into 'gitskillls'...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (3/3), done.
```

