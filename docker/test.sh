docker buildx create --use
docker buildx build -f docker/Dockerfile --push --cache-to type=local,mode=max,dest=docker_cache/cleanrl_base --cache-from type=local,src=docker_cache/cleanrl_base --platform linux/arm64,linux/amd64 -t vwxyzjn/cleanrl-base:latest .
docker buildx build -f docker/Dockerfile --push --platform linux/arm64,linux/amd64 -t vwxyzjn/cleanrl-base:latest .

docker -H ssh://me@my-arm-instance info
docker buildx create --name remote --use
docker buildx create --name remote --append ssh://costa@gpu