#!/bin/bash

clean_notebooks() {
    _BUCKET="$2"
    # Loop through each file in the path ($1/* pulls each file in the path)
    for i in $1/*
    do
        # If the current file is a directoy, 
        # then run the function again starting with the new directory
        if [ -d "$i" ]; then
            clean_notebooks "$i" "$_BUCKET"
        # If the file is indeed a file, then run nbconvert to clear output
        # inpace will replace update the file using the same name
        elif [ -f "$i" ]; then
            if [ ${i: -6} == ".ipynb" ]; then
                jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$i"
                # This will set the kernel name. All of the notebooks for this course
                # have the incorrect kernel chosen. This uses jq to replace the kernel with a working option
                kernel='conda_pytorch_p310'
                # jq '.metadata.kernelspec.name |= "'$kernel'"' $i > tempfile
                jq '.metadata.kernelspec.name |= "'$kernel'" | .metadata.kernelspec.display_name |= "'$kernel'"' $i > tempfile
                if [ $(jq '.metadata.kernelspec.name | contains("'$kernel'")' tempfile) == true ]; then mv tempfile $i; fi
            fi
        fi
    done
}
#Removing output from the notebooks
FILES=/home/ec2-user/SageMaker
clean_notebooks $FILES $1

# https://tldp.org/LDP/abs/html/comparison-ops.html
# -e file exists
# -f file is a regular file (not a directory or device file)
# -s file is not zero size
# -d file is a directory
# -r file has read permission (for the user running the test)
# -w file has write permission (for the user running the test)
# -x file has execute permission (for the user running the test)
# -z string is null, that is, has zero length
# sed -i 's/\"python3\"/\"conda_pytorch_p39\"/g' SageMaker/solution/MLUMLA-EN-M2-Lab4-solution.ipynb
