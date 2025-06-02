# v1.0.6
## Updates to mod 3 lab 4 to fix requirements
- Several library versions were set to >= in the requirements.txt file causing issues as the latest version of the libraries do not run successfully on the current version of SageMaker.

# v1.0.5
## Updates to Module 3 lab 2 to fix errors in pip install command
# v1.0.4
## All labs updated to remove NLTK
- NLTK has a security vunlerability in how it handels pickle files
- To address this NLTK was removed proactivly instead of waiting for the library to be updated
# v1.0.3
## Lab update to new pin
  - All labs are updated to the new SageMaker Notebook pin and tested for functionality
# v1.0.2
## Updates to address SageMaker AMI pinning
- Added prefix parameter to all lab.template files
- Added notebookinstancename to all instance launches
- This should allow us to pin the SageMaker AMI until we have tested new kernel versions
# v1.0.1
## Updates to labs based on customer feedback
# v1.0.0
## Initial version launch
- All labs functional
- See `HELP-KERNEL-ERRORS.md` if you are troubleshooting a kernel issue
