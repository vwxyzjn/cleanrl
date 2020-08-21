# pip install boto3
import boto3
import re
client = boto3.client('batch')

print("creating job queue")
response = client.create_job_queue(
    jobQueueName='cleanrl',
    state='ENABLED',
    priority=100,
    computeEnvironmentOrder=[
        {
            'order': 100,
            'computeEnvironment': 'cleanrl'
        }
    ]
)
print(response)
print("job queue created \n=============================")

# print("creating on demand job queue")
# response = client.create_job_queue(
#     jobQueueName='cleanrl_ondemand',
#     state='ENABLED',
#     priority=101,
#     computeEnvironmentOrder=[
#         {
#             'order': 100,
#             'computeEnvironment': 'cleanrl_ondemand'
#         }
#     ]
# )
# print(response)
# print("on demand job queue created \n=============================")


print("creating job definition")
response = client.register_job_definition(
    jobDefinitionName='cleanrl',
    type='container',
    containerProperties={
        'image': 'vwxyzjn/cleanrl:latest',
        'vcpus': 1,
        'memory': 1000,
        'privileged': True,
    },
    retryStrategy={
        'attempts': 3
    },
    timeout={
        'attemptDurationSeconds': 1800
    }
)
print(response)
print("job definition created \n=============================")


print("creating job queue")
response = client.create_job_queue(
    jobQueueName='cleanrl_gpu_large_memory',
    state='ENABLED',
    priority=100,
    computeEnvironmentOrder=[
        {
            'order': 100,
            'computeEnvironment': 'cleanrl_gpu_large_memory'
        }
    ]
)
print(response)
print("job queue created \n=============================")

print("creating job queue")
response = client.create_job_queue(
    jobQueueName='cleanrl_gpu',
    state='ENABLED',
    priority=100,
    computeEnvironmentOrder=[
        {
            'order': 100,
            'computeEnvironment': 'cleanrl_gpu'
        }
    ]
)
print(response)
print("job queue created \n=============================")
