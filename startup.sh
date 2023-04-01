# run it as . startup.sh ( not using bash or sh )

module load conda3/latest 
module load gcc/9.2.0 cuda/11.7

conda init bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/n/app/anaconda/v3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/n/app/anaconda/v3/etc/profile.d/conda.sh" ]; then
        . "/n/app/anaconda/v3/etc/profile.d/conda.sh"
    else
        export PATH="/n/app/anaconda/v3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate dmasif