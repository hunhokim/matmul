# Execution environment

Run commands that need access to the NVIDIA GPU driver outside the sandbox
(request `require_escalated`), including `nvidia-smi` and CUDA programs that
execute GPU work. The user has authorized this for GPU driver access.

Keep editing, CMake configuration, compilation, and checks that do not access
the GPU driver inside the sandbox. Do not escalate ordinary development
commands merely because this is a CUDA project.
