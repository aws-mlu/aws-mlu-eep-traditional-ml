# Kernel issues

There are 20+ Jupyter Notebook files for this course. Amazon SageMaker
deprecates kernels periodically. To make it easier to manage the lab kernel, 
the `clean_notebook.sh` file includes a line that sets the kernel.

The `clean_notebook.sh` file is located in the `aws-tc-largeobjects` folder/bucket.

The kernel is __ALWAYS__ overwritten by this script. You must update the script
to change the kernel. Updating the script will update the kernel for all labs.

`jq` is used to perform the update.

# Library issues
These labs run in Amazon SageMaker. Specific libraries need to be installed. 
To handel the installs each Notebook runs a unique `requirements.txt` file.
This means that the Notebook runs `!pip install ...` command.

The requirements are not part of the LifeCycleConfiguration created by
the CloudFormation template because LifeCycleConfiguration's can only run for
5 minutes before timing out. Many of the library installs take longer.

Make sure the `requirements.txt` file for the lab (in the `labfolder/SageMakerFiles.zipsrc` 
folder) is correct.

