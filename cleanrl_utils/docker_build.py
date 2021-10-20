import docker
client = docker.from_env()

client.images.build(
    path = './project/src/app/target/',
    dockerfile = '../../../Dockerfile/Dockerfile.name',
    tag='image:version',
)