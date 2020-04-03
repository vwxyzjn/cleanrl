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
        'image': 'vwxyzjn/cleanrl_shared_memory:latest',
        'vcpus': 1,
        'memory': 1000,
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
