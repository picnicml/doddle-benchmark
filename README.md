# doddle-benchmark

Benchmarking the [doddle-model](https://github.com/picnicml/doddle-model) implementations, against [Scikit-Learn](http://scikit-learn.org/stable/index.html).

All experiments ran multiple times, using *sbt-jmh*, for all implementations and with fixed hyperparameters, selected in a way such that models yielded similar performance.

## Implemented Algorithms

*   Linear Regression
*   Clustering **in-progress*

## Setup

#### Setup [doddle-model](https://github.com/picnicml/doddle-model)

To run the tests locally you will need to publish a local snapshot version of the repository.

    git clone https://github.com/picnicml/doddle-model.git
    cd doddle-model
    sbt publishLocal

Ensure the published version matches the version contained within the `project/Dependencies.scala` file.


#### Test Data

All test data will need to be downloaded using [Scikit-Learn], to do this simply run the Python script located in the resources folder called **.

    cd src/main/resources
    pip install -r requirements.txt
    python AcquireDataset.py [dataset as argument]

By default the data dir is set to a folder named 'scikit_learn_data' your home folder.

The baseline dataset that are already may not be substantially large enough to for benchmarking therefor a second script **RunGenerator.py** to make either classification or regression dataset. Edit the parameters in the script and then run. The parameters that are defined are the default used to benchmark the two implementations.

# Linear Regression

Dataset Name:   **Regression_data.csv**
Sample Size:    100,000

Run benchmarks - ```jmh:run -bm AverageTime -i 20 -wi 20 -f1 -t1 .*JMH_Linear*```

# Logistic Regression

Dataset Name:   **Logistic_data.csv**
Sample Size:    100,000

Run benchmarks - ```jmh:run -bm AverageTime -i 20 -wi 20 -f1 -t1 .*JMH_Logistic*```

# Softmax Classifier

Dataset Name:   **Softmax_data.csv**
Sample Size:    100,000

Run benchmarks - ```jmh:run -bm AverageTime -i 20 -wi 20 -f1 -t1 .*JMH_Softmax*```

##### Benchmark modes

Available modes are: [Throughput, AverageTime, SampleTime, SingleShotTime, All]

    -bm <mode>

##### Getting Machine Specs

To obtain your machines specs run the following command:

    cat /proc/cpuinfo
