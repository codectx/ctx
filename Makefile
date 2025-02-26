run:
	go run cmd/main.go

test:
	go test -v ./...
	# go test -v -run TestGetRelFname 

build:
	go build -o ctx cmd/main.go

demo:
	go run cmd/main.go > ctx.map

aider:
	rm -rf ./.aider.tags.cache.v3
	# source /code/aider/aider/.venv/bin/activate
	export PIP_DISABLE_PIP_VERSION_CHECK=1
	PYTHONPATH=/code/aider/aider python3 -m aider.main --show-repo-map > aider.map
