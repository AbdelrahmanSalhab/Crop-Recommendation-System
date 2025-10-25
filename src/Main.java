import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import java.util.Random;
import java.util.Scanner;
import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.text.DecimalFormat;

public class Main {
    private static final DecimalFormat df = new DecimalFormat("#.####");
    private static J48 finalClassifier;
    private static Instances trainingData;

    // Configuration constants
    private static final String DEFAULT_DATASET_PATH = "/Users/abdelrahman/Downloads/Crop_recommendation.csv";
    private static final String[] J48_OPTIONS = {"-C", "0.25", "-M", "2"};

    // Input validation ranges
    private static final double[] NITROGEN_RANGE = {0, 200};
    private static final double[] PHOSPHORUS_RANGE = {0, 150};
    private static final double[] POTASSIUM_RANGE = {0, 100};
    private static final double[] TEMPERATURE_RANGE = {0, 50};
    private static final double[] HUMIDITY_RANGE = {0, 100};
    private static final double[] PH_RANGE = {0, 14};
    private static final double[] RAINFALL_RANGE = {0, 500};

    public static void main(String[] args) {
        try {
            System.out.println("=== CROP RECOMMENDATION SYSTEM USING DECISION TREES ===\n");

            // Load dataset
            String datasetPath =  DEFAULT_DATASET_PATH;
            trainingData = loadDataset(datasetPath);
            System.out.println("Dataset loaded successfully from: " + datasetPath);
            System.out.println("Number of instances: " + trainingData.numInstances());
            System.out.println("Number of attributes: " + trainingData.numAttributes());
            System.out.println("Features: N, P, K, temperature, humidity, ph, rainfall");
            System.out.println("Classes: " + getClassLabels(trainingData));
            System.out.println();

            // Perform 5-fold cross-validation
            System.out.println("=== PERFORMING 5-FOLD CROSS-VALIDATION ===");
            performCrossValidation(trainingData);

            // Train final classifier on full dataset
            System.out.println("\n=== TRAINING FINAL CLASSIFIER ===");
            finalClassifier = createJ48Classifier();
            finalClassifier.buildClassifier(trainingData);

            System.out.println("Final Decision Tree Model:");
            System.out.println(finalClassifier);

            // Visualize the tree
            visualizeDecisionTree(finalClassifier, trainingData);

            // Interactive prediction system
            runInteractivePrediction();

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Creates a J48 classifier with predefined options
     */
    private static J48 createJ48Classifier() throws Exception {
        J48 classifier = new J48();
        // -C = confidence factor (25% - controls pruning aggressiveness)
        // -M = minimum instances per leaf (2 - prevents overfitting)
        classifier.setOptions(J48_OPTIONS);
        return classifier;
    }

    /**
     * Loads the crop dataset from CSV file
     */
    private static Instances loadDataset(String filename) throws Exception {
        File file = new File(filename);
        if (!file.exists()) {
            throw new FileNotFoundException("Dataset file not found: " + filename);
        }

        CSVLoader loader = new CSVLoader();
        loader.setSource(file);
        Instances data = loader.getDataSet();

        if (data.numInstances() == 0) {
            throw new IllegalArgumentException("Dataset is empty!");
        }

        // Convert string labels to nominal
        StringToNominal filter = new StringToNominal();
        filter.setAttributeRange("last");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // Set class attribute (last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    /**
     * Performs 5-fold cross-validation and calculates metrics
     */
    private static void performCrossValidation(Instances data) throws Exception {
        Random random = new Random(1);
        int numFolds = 5;

        // Shuffle the data
        Instances shuffledData = new Instances(data);
        shuffledData.randomize(random);

        double totalAccuracy = 0;
        double totalPrecision = 0;
        double totalRecall = 0;

        System.out.println("Fold\tAccuracy\tPrecision\tRecall");
        System.out.println("----\t--------\t---------\t------");

        for (int fold = 0; fold < numFolds; fold++) {
            // Create training and testing sets for this fold
            Instances trainSet = shuffledData.trainCV(numFolds, fold);
            Instances testSet = shuffledData.testCV(numFolds, fold);

            // Build classifier
            J48 classifier = createJ48Classifier();
            classifier.buildClassifier(trainSet);

            // Evaluate on test set
            Evaluation eval = new Evaluation(trainSet);
            eval.evaluateModel(classifier, testSet);

            // Calculate metrics
            double accuracy = eval.pctCorrect();
            double precision = calculateWeightedPrecision(eval);
            double recall = calculateWeightedRecall(eval);

            totalAccuracy += accuracy;
            totalPrecision += precision;
            totalRecall += recall;

            System.out.println((fold + 1) + "\t" + df.format(accuracy) + "%\t\t" +
                    df.format(precision) + "\t\t" + df.format(recall));
        }

        // Print averages
        System.out.println("----\t--------\t---------\t------");
        System.out.println("Avg\t" + df.format(totalAccuracy / numFolds) + "%\t\t" +
                df.format(totalPrecision / numFolds) + "\t\t" +
                df.format(totalRecall / numFolds));
    }

    /**
     * Calculates weighted precision from evaluation
     */
    private static double calculateWeightedPrecision(Evaluation eval) {
        double weightedPrecision = 0;
        double totalInstances = eval.numInstances();

        for (int i = 0; i < trainingData.classAttribute().numValues(); i++) {
            double classInstances = eval.numTruePositives(i) + eval.numFalseNegatives(i);
            double classPrecision = eval.precision(i);
            if (!Double.isNaN(classPrecision)) {
                weightedPrecision += (classInstances / totalInstances) * classPrecision;
            }
        }
        return weightedPrecision;
    }

    /**
     * Calculates weighted recall from evaluation
     */
    private static double calculateWeightedRecall(Evaluation eval) {
        double weightedRecall = 0;
        double totalInstances = eval.numInstances();

        for (int i = 0; i < trainingData.classAttribute().numValues(); i++) {
            double classInstances = eval.numTruePositives(i) + eval.numFalseNegatives(i);
            double classRecall = eval.recall(i);
            if (!Double.isNaN(classRecall)) {
                weightedRecall += (classInstances / totalInstances) * classRecall;
            }
        }
        return weightedRecall;
    }

    /**
     * Visualizes the decision tree in a GUI window
     */
    private static void visualizeDecisionTree(J48 classifier, Instances data) {
        try {
            TreeVisualizer tv = new TreeVisualizer(null, classifier.graph(), new PlaceNode2());

            JFrame frame = new JFrame("Crop Recommendation Decision Tree");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setSize(1000, 700);
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(tv, BorderLayout.CENTER);

            // Add control panel
            JPanel controlPanel = new JPanel();
            controlPanel.setLayout(new FlowLayout());
            frame.getContentPane().add(controlPanel, BorderLayout.SOUTH);

            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            System.out.println("Decision tree visualization opened in new window!");

        } catch (Exception e) {
            System.err.println("Error creating tree visualization: " + e.getMessage());
        }
    }

    /**
     * Interactive prediction system for users to input environmental parameters
     */
    private static void runInteractivePrediction() {
        System.out.println("\n=== INTERACTIVE CROP RECOMMENDATION SYSTEM ===");
        System.out.println("Enter environmental parameters to get crop recommendations:");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                try {
                    System.out.println("\n--- Enter Environmental Parameters ---");

                    double nitrogen = getValidatedInput(scanner, "Nitrogen content (N) in mg/kg",
                            "[typical range: 60-100]", NITROGEN_RANGE);

                    double phosphorus = getValidatedInput(scanner, "Phosphorus content (P) in mg/kg",
                            "[typical range: 35-60]", PHOSPHORUS_RANGE);

                    double potassium = getValidatedInput(scanner, "Potassium content (K) in mg/kg",
                            "[typical range: 15-45]", POTASSIUM_RANGE);

                    double temperature = getValidatedInput(scanner, "Average temperature in Â°C",
                            "[typical range: 15-30]", TEMPERATURE_RANGE);

                    double humidity = getValidatedInput(scanner, "Average humidity in %",
                            "[typical range: 50-85]", HUMIDITY_RANGE);

                    double ph = getValidatedInput(scanner, "Soil pH value",
                            "[typical range: 5.0-8.0]", PH_RANGE);

                    double rainfall = getValidatedInput(scanner, "Rainfall in mm",
                            "[typical range: 60-300]", RAINFALL_RANGE);

                    // Create instance for prediction
                    Instance newInstance = createPredictionInstance(nitrogen, phosphorus, potassium,
                            temperature, humidity, ph, rainfall);

                    // Make prediction
                    double prediction = finalClassifier.classifyInstance(newInstance);
                    String recommendedCrop = trainingData.classAttribute().value((int) prediction);

                    // Get confidence distribution
                    double[] distribution = finalClassifier.distributionForInstance(newInstance);

                    System.out.println("\n=== RECOMMENDATION RESULT ===");
                    System.out.println("Recommended Crop: " + recommendedCrop.toUpperCase());
                    System.out.println("Confidence: " + df.format(distribution[(int) prediction] * 100) + "%");

                    System.out.println("\nConfidence distribution for all crops:");
                    for (int i = 0; i < trainingData.classAttribute().numValues(); i++) {
                        String cropName = trainingData.classAttribute().value(i);
                        System.out.println("  " + cropName + ": " + df.format(distribution[i] * 100) + "%");
                    }

                    System.out.print("\nWould you like to make another prediction? (y/n): ");
                    String choice = scanner.next();
                    if (!choice.toLowerCase().startsWith("y")) {
                        break;
                    }

                } catch (Exception e) {
                    System.err.println("Error making prediction: " + e.getMessage());
                    scanner.nextLine(); // Clear input buffer
                }
            }
        }

        System.out.println("Thank you for using the Crop Recommendation System!");
    }

    /**
     * Gets validated input from user within specified range
     */
    private static double getValidatedInput(Scanner scanner, String paramName, String rangeInfo, double[] range) {
        while (true) {
            System.out.print(paramName + " " + rangeInfo + ": ");
            try {
                double value = scanner.nextDouble();
                if (value >= range[0] && value <= range[1]) {
                    return value;
                } else {
                    System.out.println("Warning: Value " + value + " is outside the valid range [" +
                            range[0] + "-" + range[1] + "]. Please enter a valid value.");
                }
            } catch (Exception e) {
                System.out.println("Invalid input. Please enter a numeric value.");
                scanner.nextLine(); // Clear invalid input
            }
        }
    }

    /**
     * Creates a new instance for prediction
     */
    private static Instance createPredictionInstance(double nitrogen, double phosphorus,
                                                     double potassium, double temperature,
                                                     double humidity, double ph, double rainfall) {
        Instance instance = new DenseInstance(trainingData.numAttributes());
        instance.setDataset(trainingData);

        instance.setValue(0, nitrogen);      // N
        instance.setValue(1, phosphorus);    // P
        instance.setValue(2, potassium);     // K
        instance.setValue(3, temperature);   // temperature
        instance.setValue(4, humidity);      // humidity
        instance.setValue(5, ph);            // ph
        instance.setValue(6, rainfall);      // rainfall
        // Class value will be predicted

        return instance;
    }

    /**
     * Gets all class labels from the dataset
     */
    private static String getClassLabels(Instances data) {
        StringBuilder labels = new StringBuilder();
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            if (i > 0) labels.append(", ");
            labels.append(data.classAttribute().value(i));
        }
        return labels.toString();
    }
}
