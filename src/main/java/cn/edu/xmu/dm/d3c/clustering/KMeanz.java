package cn.edu.xmu.dm.d3c.clustering;

import java.util.HashMap;
import java.util.List;
import java.util.Random;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class KMeanz extends RandomizableClusterer implements NumberOfClustersRequestable, WeightedInstancesHandler {
    private static final long serialVersionUID = 1L;

    private int m_NumClusters = 0;

    private Instances m_ClusterCentroids;

    protected DistanceFunction m_DistanceFunction = (DistanceFunction)new EuclideanDistance();

    private boolean m_PreserveOrder = true;

    private int m_Iterations = 0;

    private int m_MaxIterations = 100;

    private int[] m_ClusterSizes;

    public KMeanz(int numClusters) {
        this.m_SeedDefault = 10;
        this.m_NumClusters = numClusters;
        setSeed(this.m_SeedDefault);
    }

    public void buildClusterer(Instances data) {}

    public void buildClusterer(Instances data, List<Integer> chooseClassifiers, List<Double> correctRateArray) throws Exception {
        this.m_Iterations = 0;
        Instances instances = new Instances(data);
        this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);
        int[] clusterAssignments = new int[instances.numInstances()];
        this.m_DistanceFunction.setInstances(instances);
        Random RandomO = new Random(getSeed());
        HashMap<Object, Object> initC = new HashMap<Object, Object>();
        DecisionTableHashKey hk = null;
        Instances initInstances = null;
        if (this.m_PreserveOrder) {
            initInstances = new Instances(instances);
        } else {
            initInstances = instances;
        }
        for (int j = initInstances.numInstances() - 1; j >= 0; j--) {
            int instIndex = RandomO.nextInt(j + 1);
            hk = new DecisionTableHashKey(initInstances.instance(instIndex), initInstances.numAttributes(), true);
            if (!initC.containsKey(hk)) {
                this.m_ClusterCentroids.add(initInstances.instance(instIndex));
                initC.put(hk, null);
            }
            initInstances.swap(j, instIndex);
            if (this.m_ClusterCentroids.numInstances() == this.m_NumClusters)
                break;
        }
        this.m_NumClusters = this.m_ClusterCentroids.numInstances();
        initInstances = null;
        boolean converged = false;
        Instances[] tempI = new Instances[this.m_NumClusters];
        while (!converged) {
            int emptyClusterCount = 0;
            this.m_Iterations++;
            converged = true;
            int k;
            for (k = 0; k < instances.numInstances(); k++) {
                Instance toCluster = instances.instance(k);
                int newC = clusterProcessedInstance(toCluster);
                if (newC != clusterAssignments[k])
                    converged = false;
                clusterAssignments[k] = newC;
            }
            this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);
            for (k = 0; k < this.m_NumClusters; k++)
                tempI[k] = new Instances(instances, 0);
            for (k = 0; k < instances.numInstances(); k++)
                tempI[clusterAssignments[k]].add(instances.instance(k));
            for (k = 0; k < this.m_NumClusters; k++) {
                if (tempI[k].numInstances() == 0) {
                    emptyClusterCount++;
                } else {
                    moveCentroid(k, tempI[k]);
                }
            }
            if (emptyClusterCount > 0) {
                this.m_NumClusters -= emptyClusterCount;
                if (converged) {
                    Instances[] t = new Instances[this.m_NumClusters];
                    int index = 0;
                    for (int m = 0; m < tempI.length; m++) {
                        if (tempI[m].numInstances() > 0)
                            t[index++] = tempI[m];
                    }
                    tempI = t;
                } else {
                    tempI = new Instances[this.m_NumClusters];
                }
            }
            if (this.m_Iterations == this.m_MaxIterations)
                converged = true;
        }
        this.m_ClusterSizes = new int[this.m_NumClusters];
        for (int i = 0; i < this.m_NumClusters; i++)
            this.m_ClusterSizes[i] = tempI[i].numInstances();
        selectClassifier(clusterAssignments, chooseClassifiers, correctRateArray);
    }

    public void selectClassifier(int[] clusterAssignments, List<Integer> chooseClassifiers, List<Double> correctRateArray) {
        int chooseID = 0;
        for (int i = 0; i < this.m_NumClusters; i++) {
            double correctRate = 0.0D;
            for (int j = 0; j < clusterAssignments.length; j++) {
                if (clusterAssignments[j] == i &&
                        correctRate < ((Double)correctRateArray.get(j)).doubleValue()) {
                    correctRate = ((Double)correctRateArray.get(j)).doubleValue();
                    chooseID = j;
                }
            }
            chooseClassifiers.add(Integer.valueOf(chooseID));
        }
    }

    protected double[] moveCentroid(int centroidIndex, Instances members) {
        double[] vals = new double[members.numAttributes()];
        for (int j = 0; j < members.numAttributes(); j++) {
            if (this.m_DistanceFunction instanceof EuclideanDistance || members.attribute(j).isNominal())
                vals[j] = members.meanOrMode(j);
        }
        this.m_ClusterCentroids.add(decideCentroid(vals, members));
        return vals;
    }

    public Instance decideCentroid(double[] vals, Instances members) {
        DenseInstance denseInstance = new DenseInstance(vals.length);
        for (int q = 0; q < vals.length; q++)
            denseInstance.setValue(q, vals[q]);
        double minDistance = Double.MAX_VALUE;
        int instanceID = 0;
        for (int i = 0; i < members.numInstances(); i++) {
            double tempDistance = myDistance((Instance)denseInstance, members.instance(i));
            if (tempDistance < minDistance) {
                minDistance = tempDistance;
                instanceID = i;
            }
        }
        return members.instance(instanceID);
    }

    private int clusterProcessedInstance(Instance instance) {
        double minDist = 2.147483647E9D;
        int bestCluster = 0;
        for (int i = 0; i < this.m_NumClusters; i++) {
            double dist = 0.0D;
            try {
                dist = myDistance(instance, this.m_ClusterCentroids.instance(i));
            } catch (IndexOutOfBoundsException ie) {
                ie.printStackTrace();
            }
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        return bestCluster;
    }

    protected double myDistance(Instance first, Instance second) {
        int errorIntersect = 0;
        for (int i = 0; i < first.numAttributes(); i++) {
            if (first.value(i) == second.value(i))
                errorIntersect++;
        }
        return (double)errorIntersect/first.numAttributes();
    }

    public int numberOfClusters() throws Exception {
        return this.m_NumClusters;
    }

    public void setNumClusters(int n) throws Exception {
        if (n <= 0)
            throw new Exception("Number of clusters must be > 0");
        this.m_NumClusters = n;
    }
}
