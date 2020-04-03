VERSION=1.4-cuda10.1-cudnn7-runtime-cleanrl-0.0.1

docker build -t vwxyzjn/cleanrl_nomad:$VERSION -t vwxyzjn/cleanrl_nomad:latest -f nomad.Dockerfile .
docker build -t vwxyzjn/cleanrl:$VERSION  -t vwxyzjn/cleanrl:latest -f Dockerfile .
docker build -t vwxyzjn/cleanrl_shared_memory:$VERSION  -t vwxyzjn/cleanrl_shared_memory:latest -f sharedmemory.Dockerfile .

docker push vwxyzjn/cleanrl:latest
docker push vwxyzjn/cleanrl_nomad:latest
docker push vwxyzjn/cleanrl_shared_memory:latest
docker push vwxyzjn/cleanrl:$VERSION
docker push vwxyzjn/cleanrl_nomad:$VERSION
docker push vwxyzjn/cleanrl_shared_memory:$VERSION
