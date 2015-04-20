import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author marcelprince
 */
public class ShotPrediction {
    // Actual coordinates values with index 0 for X and index 1 for Y
    public int[] coordinates;
    // 0 for missed shot and 1 for made shot
    public int shot;
    // Normalize shot values
    public double[] normal;
    
    //Delimiter used in CSV file
    private static final String COMMA_DELIMITER = ",";
	
    //Shot attributes index
    private static final int ETYPE = 13;
    private static final int RESULT = 27;
    private static final int TYPE = 29;
    private static final int X = 30;
    private static final int Y = 31;
    
    public ShotPrediction(int[] coordinates, double[] normal, int shot) {
        this.shot = shot;
	this.coordinates = coordinates;
	this.normal = normal;
    }
    
    private static void readData(String shotsPath, ArrayList<ShotPrediction> shots) {
        // Get Shots
        BufferedReader fileReader = null;
        try {
            String line;
     
            //Create the file reader
            fileReader = new BufferedReader(new FileReader(shotsPath));
            
            //Read the CSV file header to skip it
            fileReader.readLine();
            
            //Read the file line by line starting from the second line
            while ((line = fileReader.readLine()) != null) {
                String[] data = line.split(COMMA_DELIMITER);
                int[] coord = new int[2];
                double[] norm = new double[2];
                int result;
                switch (data[ETYPE]) {
                    case "shot":
                        coord[0] = Integer.parseInt(data[X]);
                        norm[0] = (coord[0] - 25.0) / 25.0;
                        coord[1] = Integer.parseInt(data[Y]);
                        if (coord[1] > 47)
                            coord[1] = 47;
                        norm[1] = (47.0 - norm[1]) / 47.0;
                        if (data[RESULT].equals("made"))
                            result = 1;
                        else
                            result = 0;
                        shots.add(new ShotPrediction(coord,norm,result));
                        break;
                    case "free throw":
                        coord[0] = 25;
                        coord[1] = 19;
                        norm[0] = 1.0;
                        norm[1] = (47.0 - 19.0)/47.0;
                        if (data[RESULT].equals("made"))
                            result = 1;
                        else
                            result = 0;
                        shots.add(new ShotPrediction(coord,norm,result));
                        break;
                }
            }
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            if (fileReader != null)
                try {
                    fileReader.close();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
        }
    }
   
    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        // Size of arrays
        int inputSize = 2;
        int outputSize = 2;
        
        // Number of Neurons, learning rate, number of training sessions
        int hiddenSize = 100;
        double learn = 0.5;
        int epochs = 150;
        
        // Initialize the arrays needed
        double[] hidden = new double[hiddenSize];
        double[] output = new double[outputSize];
        double[] error = new double[outputSize];
        double[][] weight1 = new double[hiddenSize][inputSize];
        double[][] weight2 = new double[outputSize][hiddenSize];
        double[] h1 = new double[hiddenSize];
        double[] h2 = new double[outputSize];
        double[] d2 = new double[outputSize];
        double[] d1 = new double[hiddenSize];
        double[][] delta2 = new double[outputSize][hiddenSize];
        double[][] delta1 = new double[hiddenSize][inputSize];
        weight1 = randomizeArray(weight1,hiddenSize,inputSize);
        weight2 = randomizeArray(weight2,outputSize,hiddenSize);
        
        // Get shots
        ArrayList<ShotPrediction> shotsTrain = new ArrayList<>();
        String fileTrain1 = System.getProperty("user.dir")+"/20061031.CHIMIA.csv";
        String fileTrain2 = System.getProperty("user.dir")+"/20061031.PHXLAL.csv";
        String fileTrain3 = System.getProperty("user.dir")+"/20061101.ATLPHI.csv";
        String fileTrain4 = System.getProperty("user.dir")+"/20061101.CHIORL.csv";
        String fileTrain5 = System.getProperty("user.dir")+"/20061101.HOUUTA.csv";
        String fileTrain6 = System.getProperty("user.dir")+"/20061101.INDCHA.csv";
        String fileTrain7 = System.getProperty("user.dir")+"/20061101.LACPHX.csv";
        String fileTrain8 = System.getProperty("user.dir")+"/20061101.MILDET.csv";
        readData(fileTrain1,shotsTrain);
        readData(fileTrain2,shotsTrain);
        readData(fileTrain3,shotsTrain);
        readData(fileTrain4,shotsTrain);
        readData(fileTrain5,shotsTrain);
        readData(fileTrain6,shotsTrain);
        readData(fileTrain7,shotsTrain);
        readData(fileTrain8,shotsTrain);
        
        // Train neural Network
        int shotsTrainSize = shotsTrain.size();
        for (int l = 0; l < epochs; l++){
            for (int n=0; n < shotsTrainSize; n++){
                ShotPrediction trainShot = shotsTrain.get(n);
                //feed forward
                for (int i=0; i < hiddenSize; i++){
                    double h = 0;
                    for (int j=0; j < inputSize; j++)
                         h += trainShot.normal[j] * weight1[i][j];
                    h1[i] = h;
                }
                for (int i=0; i < hiddenSize; i++){
                    hidden[i] = sigmoid(h1[i]);
                }
                for (int i=0; i < outputSize; i++){
                    double h = 0;
                    for (int j=0; j < hiddenSize; j++)
                         h += hidden[j] * weight2[i][j];
                    h2[i] = h;
                }
                for (int i=0; i < outputSize; i++){
                    output[i] = sigmoid(h2[i]);
                }

                // Calculate error
                for (int i=0; i < outputSize; i++){
                    if (i == trainShot.shot)
                        error[i] = 1 - output[i];
                    else
                        error[i] = 0 - output[i];
                }

                // BackPropagate
                for (int i=0; i < outputSize; i++){
                    d2[i] = sigmoidPrime(h2[i]) * error[i];
                }
                for (int j=0; j < hiddenSize; j++){
                    double sum = 0;
                    for (int i=0; i < outputSize; i++)
                        sum += d2[i] * weight2[i][j];
                    d1[j] = sigmoidPrime(h1[j]) * sum;
                }

                // Calculate the deltas
                for (int i=0; i < outputSize; i++){
                    for (int j=0; j < hiddenSize; j++)
                        delta2[i][j] = learn * d2[i] * hidden[j];
                }
                for (int i=0; i < hiddenSize; i++){
                    for (int j=0; j < inputSize; j++)
                        delta1[i][j] = learn * d1[i] * trainShot.normal[j];
                }

                // Modify the weight matrices;
                for (int i=0; i < outputSize; i++){
                    for (int j=0; j < hiddenSize; j++)
                        weight2[i][j] += delta2[i][j];
                }
                for (int i=0; i < hiddenSize; i++){
                    for (int j=0; j < inputSize; j++)
                        weight1[i][j] += delta1[i][j];
                }
            }
        }
        
        ArrayList<ShotPrediction> shotsTest = new ArrayList<>();
        String fileTest1 = System.getProperty("user.dir")+"/20061101.NOKBOS.csv";
        String fileTest2 = System.getProperty("user.dir")+"/20061101.PORSEA.csv";
        readData(fileTest1,shotsTest);
        readData(fileTest2,shotsTest);
        
        // Test Neural Networks
        int count = 0;
        int shotsTestSize = shotsTest.size();
        for (int n=0; n < shotsTestSize; n++){
            ShotPrediction testShot = shotsTest.get(n);
            for (int i=0; i < hiddenSize; i++){
                double h = 0;
                for (int j=0; j < inputSize; j++)
                     h += testShot.normal[j] * weight1[i][j];
                h1[i] = h;
            }
            for (int i=0; i < hiddenSize; i++){
                hidden[i] = sigmoid(h1[i]);
            }
            for (int i=0; i < outputSize; i++){
                double h = 0;
                for (int j=0; j < hiddenSize; j++)
                     h += hidden[j] * weight2[i][j];
                h2[i] = h;
            }
            for (int i=0; i < outputSize; i++){
                output[i] = sigmoid(h2[i]);
            }

            double max = 0.0;
            int index = 0;
            for (int i=0; i < outputSize; i++){
                if (output[i] > max){
                    max = output[i];
                    index = i;
                }
            }

            if (index == testShot.shot)
                count = count + 1;
        }

        System.out.println("Total Shots Predicted: " + count);
        System.out.println("Total Shots Tested: " + shotsTestSize);
        double accuracy;
        accuracy = (double) (count * 100) / (double) shotsTestSize;
        System.out.println("Accuracy: " + accuracy);
    }
    
    public static double[][] randomizeArray(double[][] aray, int rows, int cols){
        for (int i=0; i < rows; i++)
            for (int j=0; j < cols; j++)
            aray[i][j] = 2*Math.random()-1;
        return aray;
    }
    
    public static double sigmoid(double h){
        return 1.0 / ( 1.0 + (Math.expm1(-h) + 1.0) );
    }
    
    public static double sigmoidPrime(double h){
        return sigmoid(h) * ( 1.0 - sigmoid(h) );
    }
    
}
