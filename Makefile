install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat Results/metrics.txt >> report.md
	echo '\n## Predicted vs. Actual' >> report.md
	echo '![Predicted vs Actual](./Results/model_results.png)' >> report.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add .
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	hf upload MingC9899/CICD-project-2 ./App --repo-type=space --commit-message="Sync App files"
	hf upload MingC9899/CICD-project-2 ./Model /Model --repo-type=space --commit-message="Sync Model"
	hf upload MingC9899/CICD-project-2 ./Data /Data --repo-type=space --commit-message="Sync Data"
	hf upload MingC9899/CICD-project-2 ./Results /Metrics --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy