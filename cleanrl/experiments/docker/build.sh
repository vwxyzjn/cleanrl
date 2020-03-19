VERSION=1.4-cuda10.1-cudnn7-runtime

docker build -t vwxyzjn/cleanrl_nomad:$VERSION -t vwxyzjn/cleanrl_nomad:latest -f nomad.Dockerfile .
docker build -t vwxyzjn/cleanrl:$VERSION  -t vwxyzjn/cleanrl:latest -f Dockerfile .

docker push vwxyzjn/cleanrl:latest
docker push vwxyzjn/cleanrl_nomad:latest
docker push vwxyzjn/cleanrl:$VERSION
docker push vwxyzjn/cleanrl_nomad:$VERSION
