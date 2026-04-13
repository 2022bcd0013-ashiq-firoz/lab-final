pipeline {
    agent any

    
    stages {

        stage('Checkout') {
            steps {
               git branch: 'main',
                    url: 'https://github.com/2022bcd0013-ashiq-firoz/lab-final.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }

        stage('DVC Pull'){
            steps{
                sh'''
                dvc pull
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . venv/bin/activate

                    mkdir -p output
                    python Script/train.py

                    mkdir -p training-artifacts-py3.11
                    cp -r output/* training-artifacts-py3.11/

                    echo "Contents of training-artifacts-py3.11/:"
                    ls -la training-artifacts-py3.11/
                '''
            }
        }

        stage('Archive Artifacts') {
            steps {
                script {
                    sh 'ls -la training-artifacts-py3.11/'
                    archiveArtifacts artifacts: 'training-artifacts-py3.11/**/*', allowEmptyArchive: false
                    stash includes: 'training-artifacts-py3.11/**/*', name: 'model-artifacts'
                    echo "Model artifacts archived successfully"
                }
            }
        }

        
    }
}