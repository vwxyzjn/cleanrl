job "cleanrl_job" {
  type        = "batch"
  datacenters = ["dc1"]
  meta {
    command = ""
  }
  parameterized {
    meta_optional = ["command", "wandb"]
  }
  group "cleanrl_group" {
    task "cleanrl_task" {
      driver = "docker"
      config {
        image = "vwxyzjn/cleanrl_nomad:1.4-cuda10.1-cudnn7-runtime"
        command = "${NOMAD_META_COMMAND}"
        # cpu_hard_limit = true
      }
      env {
        WANDB = "${NOMAD_META_WANDB}"
        MKL_NUM_THREADS=1
        OMP_NUM_THREADS=1
      }
      resources {
        cpu = 2000
        memory = 1000
      }
    }
    restart {
      attempts = 0
      mode = "fail"
    }
  }
  reschedule {
    attempts = 0
  }
}
