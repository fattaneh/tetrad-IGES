package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

class KeyMB {

	public final int n_a;
	public final int n_d;
	public final int n_r;


	public KeyMB(final int n_a, final int n_d, final int n_r) {
		this.n_a = n_a;
		this.n_d = n_d;
		this.n_r = n_r;
	}
	@Override
	public boolean equals (final Object O) {
		if (!(O instanceof KeyMB)) return false;
		if (((KeyMB) O).n_a != n_a) return false;
		if (((KeyMB) O).n_d != n_d) return false;
		if (((KeyMB) O).n_r != n_r) return false;
		return true;
	}
	 @Override
	 public int hashCode() {
		 return this.n_a ^ this.n_d ^ this.n_r ;
	 }
	 public String print(KeyMB key){
		return "("+key.n_a +", "+ key.n_d +", "+ key.n_r + ")";
	 }

}
public class TestISFGES_MB_LOOCV {
	public static void main(String[] args) {
		
		String pathToFolder = "./UCI/";
		String dataName = "SPECT.train";
		String pathToData = pathToFolder + dataName + ".csv";
		String target = "y";

		
		// Read in the data
		DataSet trainDataOrig = readData(pathToData);

		// set parameter priors
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		
		// learn the population model using all training data
		Graph graphP = BNlearn_pop(trainDataOrig, samplePrior, structurePrior);
		System.out.println("Pop graph:" + graphP.getEdges());
		
		// Log the results
		PrintStream logFile;
		try {
			File dir = new File( pathToFolder + "/IGES/MB/" + dataName + "/PESS" + samplePrior);
			dir.mkdirs();
			String outputFileName = dataName + "PESS" + samplePrior +"_log.txt";
			File fileAUC = new File(dir, outputFileName);
			logFile = new PrintStream(new FileOutputStream(fileAUC));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		logFile.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		logFile.println("Pop graph:" + graphP.getEdges());

		System.out.println("PESS = " + samplePrior);
		logFile.println("PESS = " + samplePrior);


		// Run for different kappa values
		for (int p = 0; p < 10; p++){

			double kappa =  p/10.0; 

			// Store the probability of the target variable using instance-specific and population-wide models
			double[] probs_is = new double[trainDataOrig.getNumRows()];
			double[] probs_pw = new double[trainDataOrig.getNumRows()];
			
			// Truth 
			int[] truth = new int[trainDataOrig.getNumRows()];

			// Compare the Markov blankets of the instance-specific models versus the population-wide model
			Map <KeyMB, Double> stats= new HashMap<KeyMB, Double>();
			
			PrintStream outForAUC, out;
			try {
				File dir = new File( pathToFolder + "/IGES/MB/" + dataName + "/PESS" + samplePrior);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ kappa + "PESS" + samplePrior +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
				
				outputFileName = dataName + "FeatureDist-Kappa"+ kappa + "PESS" + samplePrior +".csv";
				File filePredisctors = new File(dir, outputFileName);
				out = new PrintStream(new FileOutputStream(filePredisctors));

			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			System.out.println("kappa = " + kappa);
			logFile.println("kappa = " + kappa);
			
			Map <String, Double> fdist= new HashMap<String, Double>();
			for (int i = 0; i < trainDataOrig.getNumColumns(); i++){
				fdist.put(trainDataOrig.getVariable(i).getName(), 0.0);
			}
			
			out.println("features, fraction of occurance in cases");
			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");

			//LOOCV over the training instances
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){

				DataSet trainData = trainDataOrig.copy();
				DataSet test = trainDataOrig.subsetRows(new int[]{i});
				trainData.removeRows(new int[]{i});

				// learn the instance-specific BN
				Graph graphI = learnBNIS(trainData, test, kappa, graphP, samplePrior);

				// compute probability distribution of the target variable using the instance-specific and population-wide model
				int targetIndex = trainData.getColumn(trainData.getVariable(target)); 
				truth[i] = test.getInt(0, targetIndex);
				
				//get the probability from IS model
				DagInPatternIterator iterator = new DagInPatternIterator(graphI);
				Graph dagI = iterator.next();
				dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
				Graph mb_i = GraphUtils.markovBlanketDag(dagI.getNode(target), dagI);
				probs_is[i]= estimation(trainData, test, (Dag) mb_i, target);
				List<Node> mb_nodes = mb_i.getNodes();
				mb_nodes.remove(mb_i.getNode(target));
				for (Node no: mb_nodes){
					fdist.put(no.getName(), fdist.get(no.getName()) + 1.0);
				}

				//get the probability from the population model
				DagInPatternIterator iteratorP = new DagInPatternIterator(graphP);
				Graph dagP = iteratorP.next();
				dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
				Graph mb_p = GraphUtils.markovBlanketDag(dagP.getNode(target), dagP);
				probs_pw[i] = estimation(trainData, test, (Dag) mb_p, target);

				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setSamplePrior(samplePrior);
				scoreI.setKAddition(kappa);
				scoreI.setKDeletion(kappa);
				scoreI.setKReorientation(kappa);
				ISFges iges = new ISFges(scoreI);				
				iges.setPopulationGraph(graphP);
				List <Node> mb_nodes_all = mb_i.getNodes();
				mb_nodes_all.addAll(mb_p.getNodes());
				Graph mb_i2 = new EdgeListGraph(mb_nodes_all);
				Graph mb_p2 = new EdgeListGraph(mb_nodes_all);
				for (Edge e : mb_i.getEdges()){
					mb_i2.addEdge(e);
				}
				for (Edge e : mb_p.getEdges()){
					mb_p2.addEdge(e);
				}
				
				
				// Markov blanket comparison (added, deleted and re-oriented nodes)
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(mb_i, mb_p);
				
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				int n_r = cmp.getEdgesReorientedFrom().size();

				KeyMB cur_key = new KeyMB(n_a, n_d, n_r);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

				outForAUC.println(test.getInt(0, targetIndex) +", " + probs_pw[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
			}
			double auroc_p = AUC.measure(truth, probs_pw);
			double auroc = AUC.measure(truth, probs_is);

			System.out.println( "AUROC_P: "+ auroc_p);
			System.out.println( "AUROC: "+ auroc);
			logFile.println( "AUROC_P: "+ auroc_p);
			logFile.println( "AUROC: "+ auroc);

			for (KeyMB k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
				logFile.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);

			}
			System.out.println("-----------------");
			logFile.println("-----------------");
			
			
			// write the statistics to the output file
			Map<String, Double> sortedfdist = sortByValue(fdist, false);
			for (String k : sortedfdist.keySet()){
				out.println(k + ", " + (fdist.get(k)/trainDataOrig.getNumRows()));
				
			}
			outForAUC.close();
			out.close();
		}
		logFile.close();
	}

	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Dag mb, String target){

		double [] probs = classify(mb, trainData, test, (DiscreteVariable) test.getVariable(target));
	
		return probs[1];
	}

	public static double[] classify(Dag mb, DataSet train, DataSet test, DiscreteVariable targetVariable) {

		List<Node> mbNodes = mb.getNodes();

		//Find the subset dataset that corresponds to the Markov blanket. 
		DataSet trainDataSubset = train.subsetColumns(mbNodes);

		//To parameterize the Bayes net we need the number of values of each variable.
		BayesPm bayesPm = new BayesPm(mb);
		List<Node> varsTrain = trainDataSubset.getVariables();

		for (int i1 = 0; i1 < varsTrain.size(); i1++) {
			DiscreteVariable trainingVar = (DiscreteVariable) varsTrain.get(i1);
			bayesPm.setCategories(mbNodes.get(i1), trainingVar.getCategories());
		}

		//Create an updater for the instantiated Bayes net.
		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(bayesPm, 1.0);
		BayesIm bayesIm = DirichletEstimator.estimate(prior, trainDataSubset);

		RowSummingExactUpdater updater = new RowSummingExactUpdater(bayesIm);

		//The subset dataset of the dataset to be classified containing
		//the variables in the Markov blanket.
		DataSet testSubset = test.subsetColumns(mbNodes);

		//Get the raw data from the dataset to be classified, the number
		//of variables, and the number of cases.
		double[] estimatedProbs = new double[targetVariable.getNumCategories()];

		List<Node> varsClassify = testSubset.getVariables();

		//For each case in the dataset to be classified compute the estimated
		//value of the target variable and increment the appropriate element
		//of the crosstabulation array.

		//Create an Evidence instance for the instantiated Bayes net
		//which will allow that updating.
		Proposition proposition = Proposition.tautology(bayesIm);

		//Restrict all other variables to their observed values in this case.
		int numMissing = 0;

		for (int testIndex = 0; testIndex < varsClassify.size(); testIndex++) {
			DiscreteVariable var = (DiscreteVariable) varsClassify.get(testIndex);

			// If it's the target, ignore it.
			if (var.equals(targetVariable)) {
				continue;
			}

			int trainIndex = proposition.getNodeIndex(var.getName());

			// If it's not in the train subset, ignore it.
			if (trainIndex == -99) {
				continue;
			}

			int testValue = testSubset.getInt(0, testIndex);

			if (testValue == -99) {
				numMissing++;
			} else {
				proposition.setCategory(trainIndex, testValue);
			}
		}

		Evidence evidence = Evidence.tautology(bayesIm);
		evidence.getProposition().restrictToProposition(proposition);
		updater.setEvidence(evidence);

		// for each possible value of target compute its probability in
		// the updated Bayes net.  Select the value with the highest
		// probability as the estimated getValue.
		int targetIndex = proposition.getNodeIndex(targetVariable.getName());


		for (int category = 0; category < targetVariable.getNumCategories(); category++) {
			double marginal = updater.getMarginal(targetIndex, category);
			estimatedProbs [category] = marginal;
		}

		return estimatedProbs;
	}


	private static Graph BNlearn_pop(DataSet trainDataOrig, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals(target)){
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);
		return knowledge;
	}
	
	private static DataSet readData(String pathToData) {
		Path trainDataFile = Paths.get(pathToData);
		char delimiter = ',';
		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		return trainDataOrig;
	}
	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
	{
		List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());

		// Sorting the list based on values
		list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
				? o1.getKey().compareTo(o2.getKey())
						: o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
						? o2.getKey().compareTo(o1.getKey())
								: o2.getValue().compareTo(o1.getValue()));
		return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));

	}
}