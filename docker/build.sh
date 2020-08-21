VERSION=1.4-cuda10.1-cudnn7-runtime-cleanrl-0.3.0

docker build -t vwxyzjn/cleanrl:$VERSION  -t vwxyzjn/cleanrl:latest -f Dockerfile .
docker push vwxyzjn/cleanrl:latest
docker push vwxyzjn/cleanrl:$VERSION

docker build -t vwxyzjn/cleanrl:m$VERSION  -t vwxyzjn/cleanrl:mlatest -f m.Dockerfile .
docker push vwxyzjn/cleanrl:mlatest
docker push vwxyzjn/cleanrl:m$VERSION
