------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
This repository accompanies the article:
"
Proximal Iteration for Nonlinear Adaptive Lasso
"
which is presently under review.
------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------


------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
Installation
------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

-- Python modules
This repository contains a requirements.txt file which may be used to create a Python virtual environment. 

One way to acomplish this is:
bash> python -m venv .venv
bash> source .venv/bin/activate
bash> python -m pip install --upgrade pip
bash> python -m pip install -r requirements.txt

-- R packages
MLGL and glmnet are accessed from R via the rpy2 package; hence you will need an installation of R with those packages installed to run the comparative simulations:
R> install.packages(c("MLGL","glmnet"))


------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
Main Library Files
------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

The proximal gradient method is primarily implemented in python/jax_nsa.py.
The hyperpriors are implemented in python/jax_hier_lib.py


------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
Reproducing Numerical Results from the article.
------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

---
Figures 3-6 may be reproduced by running the script "parrun.sh".
This involves running about 500 simulations in parallel over 10 processes, which will take several days.
Make sure you :
1) Have installed gnu-parallel (maybe via "sudo apt install parallel").
2) Have given "parrun.sh" execution permission (maybe via "chmod +x parrun.sh").
3) Are executing the script from the root directory of this respository.
4) Have your virtual environment activated.

---
Unfortunately, the two case studies of section 6.4/6.5 involve datasets which we are not able to release publically. 
Therefore, our scripts create synthetic versions of these datasets to illustrate our methodology.

---
An analogue to Figure 7 but with synthetic data may be created by running:
bash> python python/mosaic_nn.py
Make sure you :
1) Are executing the script from the root directory of this respository.
2) Have your virtual environment activated.

---
An analogue to Figure 8 but with synthetic data may be created by running the script "hcr_parrun.sh".
Make sure you :
1) Have installed gnu-parallel (maybe via "sudo apt install parallel").
2) Have given "hcr_parrun.sh" execution permission (maybe via "chmod +x hcr_parrun.sh").
3) Are executing the script from the root directory of this respository.
4) Have your virtual environment activated.
