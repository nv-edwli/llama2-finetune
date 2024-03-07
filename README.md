# NVIDIA AI Workbench: Introduction
This is an [NVIDIA AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) example Project that demonstrates how to fine-tune a Llama 2 large language model (LLM) on a custom dataset using NeMo Framework. Users in the [AI Workbench Beta Program](https://developer.nvidia.com/ai-workbench-beta) can get up and running with this Project in minutes. Please note the project requirements:

* Operating System: Ubuntu 22.04, Windows (WSL2), MacOS 12+
* CPU requirements: None, (tested with Intel&reg; Xeon&reg; Platinum 8380 CPU @ 2.30GHz)
* GPU requirements: **Minimum 1x NVIDIA A100-80GB**

### Project Description
Llama 2 has gained traction as a robust, powerful family of Large Language Models that can provide compelling responses on a wide range of tasks. While the base 7B, 13B, and 70B models serve as a strong baseline for multiple downstream tasks, they can lack in domain-specific knowledge of proprietary or otherwise sensitive information. Fine-tuning is often used as a means to update a model for a specific task or tasks to better respond to domain-specific prompts. These notebooks walk through downloading and configuring the Llama 2 model from HuggingFace, preparing a custom dataset, and fine-tuning the pretrained base model against this new dataset. The 7B model has been selected by default in this project, but it is also configurable with the 13B and 70B versions of the model, depending on your compute resources and constraints. 

* ```llama2-lora-ft.ipynb```: This notebook provides a sample workflow for fine-tuning the Llama 2 parameter base model for extractive Question-Answering on the SQuAD dataset using Low-Rank Adaptation Fine-tuning (LoRA), a popular parameter-efficient fine-tuning method. 

* ```llama2-ptuning.ipynb```: This notebook provides a sample workflow for fine-tuning the Llama 2 parameter base model for extractive Question-Answering on a custom dataset using customized prompt formattings and a p-tuning method.

Have questions? Please direct any issues, fixes, suggestions, and discussion on this project to the DevZone Members Only Forum thread [here](https://forums.developer.nvidia.com/t/support-workbench-example-project-llama-2-finetune/278375). 

# Quickstart (Llama-2 7B)

### Prerequisites
1. The NeMo Frameworks training container is used by this example project. This container is in General Availability and is open to the public. Sign up for access [here](https://developer.nvidia.com/nemo-framework/join). Once you have access, AI Workbench will use your NGC API Key integration to pull this container; ensure you have this integration configured on AI Workbench. 

2. AI Workbench will prompt you to provide a few pieces of information before running any apps in this project. Ensure you have this information ready. 
   
   * The location for the model on the underlying host system. AI Workbench uses a volume mount for the ``/project/models`` directory, so the user will need to specify what location on the underlying host system the ``/project/models`` in the container will point to, and thus where on the underlying host machine the model weights will be downloaded.
   * The Hugging Face username. This is used as part of the authentication credentials to pull the Llama 2 model weights to the project. 
   * The Hugging Face API Key. This is used as part of the authentication credentials to pull the Llama 2 model weights to the project. Llama 2 is a gated model that is available for commercial use. To be able to download the model, [submit a request on Meta's portal](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) for access to all models in the Llama family. Please note that your HuggingFace account email address MUST match the email you provide on the Meta website, or your request will not be approved.

### Tutorial (Desktop App)

If you do not NVIDIA AI Workbench installed, first follow the installation instructions for AI Workbench [here](https://developer.nvidia.com/ai-workbench-beta/). Then, 

1. Fork this Project to your own GitHub namespace and copying the link

   ```
   https://github.com/[your_namespace]/<project_name>
   ```
   
2. Open NVIDIA AI Workbench. Select a location to work in. 
   
3. Clone this Project onto your desired machine by selecting "Clone Project" and providing the GitHub link.
   
4. Wait for the project to build. 
   
5. When the build completes, set the following configurations.

   * `Environment` &rarr; `Mounts` &rarr; `Configure`. Specify the file path of the mount, eg. where the Llama 2 model will live on your **host** machine.
   
      eg. if your downloaded Llama2 model directory resides in your home path, enter ```/home/[user]```

   * `Environment` &rarr; `Secrets` &rarr; `Configure`. Specify the Hugging Face username and API Key secrets.

6. On the top right of the window, select **Setup** to set up fine-tuning for the 7B model. _This will take several minutes to run_ as the model is pulled and converted to ``.nemo`` format.

   * If you would like to use a different model instead of the default 7b, set ``Environment`` &rarr; ``Variables`` &rarr; ``LLAMA2_SIZE`` to "7b" (default), "13b", or "70b". 

7. Upon completion, select the _dropdown_ from the top right of the window. Select **Jupyterlab**. Verify ```llama-2-[XX]b.nemo``` exists under the ```models```directory of the project.

8. Navigate to the `code/llama-2-[XX]b` directory of the project. Then, open your fine-tuning notebook of choice and get started. Happy coding!

### Tutorial (CLI-Only)
Some users may choose to use the **CLI tool only** instead of the Desktop App. If you do not NVIDIA AI Workbench installed, first follow the installation instructions for AI Workbench [here](https://developer.nvidia.com/ai-workbench-beta/). Then, 
1. Fork this Project to your own GitHub namespace and copying the link

   ```
   https://github.com/[your_namespace]/<project_name>
   ```
   
2. Open a shell and activating the Context you want to clone into by

   ```
   $ nvwb list contexts
   
   $ nvwb activate <desired_context>
   ```
   
3. Clone this Project onto your desired machine by running

   ```
   $ nvwb clone project <your_project_link>
   ```
   
4. Open the Project by

   ```
   $ nvwb list projects
   
   $ nvwb open <project_name>
   ```
   
5. Start **Setup** by

   ```
   $ nvwb start setup
   ```

   * Specify the file path of the mount, eg. where the Llama 2 model will live on your **host** machine.
   
      eg. if your downloaded Llama2 model directory resides in your home path, enter ```/home/[user]```

   * Specify the Hugging Face username and API Key secrets.

6. Start **Jupyterlab** by

   ```
   $ nvwb start jupyterlab
   ```

7. Verify ```llama-2-[XX]b.nemo``` exists under the ```models```directory of the project.

8. Navigate to the `code/llama-2-[XX]b` directory of the project. Then, open your fine-tuning notebook of choice and get started. Happy coding!

---
**Tip:** Use ```nvwb help``` to see a full list of NVIDIA AI Workbench commands. 

---

## Tested On
This notebook has been tested with a 1x NVIDIA A100-80gb GPU system and the Beta version of NVIDIA AI Workbench: ```nvwb 0.13.14 (stable; linux; amd64; go1.21.3; Wed Jan 10 20:09:13 UTC 2024)```

# License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/nv-edwli/llama-2-finetune/blob/main/LICENSE.txt)

This project may utilize additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 
