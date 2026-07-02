This project is built based on:

- context-private/01-Corbetta-2023-PIML-PINN-LBM-operator-10189_2023_Article_267.pdf
- and then expanded based on   context-private/01-conti01-Cabbana-2025-enhancing-lattice-kinetic-schemes-for-fluid-dynamics-with-lattice-equivariant-neural-networks.pdf .

we are still missing the implementation of free turbulance validation described in:

- context-private/01-conti01-Cabbana-2025-enhancing-lattice-kinetic-schemes-for-fluid-dynamics-with-lattice-equivariant-neural-networks.pdf .

implement the free turbulance validation.

you may refer to:

- ../workspace-master.course.block05-ML4PhA/presentation which a presentation summarizing the current result of this repository.



# Further steps
actually we have trained model here /home/tai/work-my-projects/workspace-master.course.block05-ML4PhA/data (in this folder you may find directory layout description md and the models) . use this
  lenn_resnet_karman_every_100_samp334_bs32_ep12000_pat2000_lr1e-3 for validation. (I will run the job officially on snellius. just make a one line command for me to invoke. here on the local laptop we can
  just make a small run for our implmentation validation and verification)

