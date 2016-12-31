import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 *
 * You must add code for the 5 methods specified below.
 *
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
    private DecTreeNode root;
    private List<String> labels; // ordered list of class labels
    private List<String> attributes; // ordered list of attributes
    private Map<String, List<String>> attributeValues; // map to ordered
    // discrete values taken
    // by attributes
    private DataSet thisData;
    /**
     * Answers static questions about decision trees.
     */
    DecisionTreeImpl() {
	// no code necessary
	// this is void purposefully
    }

    /**
     * Build a decision tree given only a training set.
     *
     * @param train: the training set
     */
    DecisionTreeImpl(DataSet train) {

	this.labels = train.labels;
	this.attributes = train.attributes;
	this.attributeValues = train.attributeValues;
	this.thisData = train;
	// TODO: add code here

	/**
	 * "In both DecisionTreeImpl methods, you are first required
	 * to build the decision tree with the training data."
	 */

	// Basically, pass in training data and start at root (-1)
	this.root = buildTree(train, train.instances,
			      train.attributes, null, -1);
    }

    /**
     * Build a decision tree given a training set then prune it using a tuning
     * set.
     *
     * @param train: the training set
     * @param tune: the tuning set
     */
    DecisionTreeImpl(DataSet train, DataSet tune) {

        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;
			this.thisData = train;
	// TODO: add code here

	/**
         * "In both DecisionTreeImpl methods, you are first required
         * to build the decision tree with the training data."
         */
	// Basically, pass in training data and start at root (-1)
        this.root = buildTree(train, train.instances,
			      train.attributes, null, -1);
	
	// Now, prune based on the tuning set
	prune(train, tune, root);
    }


    /**
     * Pruning from CS 540 slides: 
     * Prune (tree T, TUNE set)
     * 1. Compute T's accuracy on TUNE, call it A(T)
     * 2. For every internal node N in T:
     *    a) New tree TN = copy of T, but prune (delete the subtree
     *       under N
     *    b) N becomes a leaf node in TN. The label is the majority
     *       vote of TRAIN examples reaching N.
     *    c) A(TN) = TN's accuracy on TUNE
     * 3. Let T* be the tree (among the TN's and T) with the largest
     * A(). Set T<-T*  prune 
     * 4. Repeat from step 1 until no more improvement available
     * Return T.
     */
    private void prune(DataSet train, DataSet tune, DecTreeNode root) {

	
	double AT = 0.0; // Per algorithm step 1
	double ATN = 0.0;
	double maxATN = -1;
	List <Integer> pruneAt = new ArrayList<Integer>();

	// List of post pruned nodes off of TUNE
	List <DecTreeNode> postPrunedNodes = new ArrayList<DecTreeNode>();
	
	// Step 1: Compute T's accuracy on TUNE (call it A(T))
	AT = accuracy(train, tune);

	// Step 2: For every internal node N in T:
	// New tree TN = copy of T, but prune (delete the subtree under N)
	// N becomes a leaf node in TN. The label is the majority vote of
	// TRAIN examples reaching N.
	//postPrunedNodes = traverse(root, postPrunedNodes);
	
	// A(TN) = TN's accuracy on TUNE
	for (int i = 0; i < root.children.size(); i++) {

	    // Recursive DFS
	    postPrunedNodes = traverse(root, postPrunedNodes);
	    ATN = accuracy(train, tune);
       
	    // New max or equal max accuracy, add to list
	    if (ATN >= maxATN) {
		maxATN = ATN;
		pruneAt.add(i);       
	    }
	    
	}

	// Step 3:
	// Loop through prune at list with pruning indices
	for (int i = 0; i < pruneAt.size(); i++) {

	    // If this isn't the root, prune at certain node
	    if (pruneAt.get(i) != 0) {
		postPrunedNodes.get(pruneAt.get(i)).terminal = true;
		postPrunedNodes.get(pruneAt.get(i)).children = null;
	    }
	}

	// Step 4: I wasn't sure if we're supposed to loop or something?
    }

    /**
     * accuracy is a helper function to prune that determines the generalized
     * accuracy as measured by a training set.
     */
    private double accuracy(DataSet train, DataSet tune) {

	/**
	 * From Wikipedia: "...the best tree is chosen by
	 * generalized accuracy as measured by a training set
	 * or cross-validation
	 */

	int accuracyHits = 0;  // See below
	double accuracy = 0.0; // Accuracy to return

	// Calculate accuracy over tune.instances
	for (int i = 0; i < tune.instances.size(); i++) {

	    // If tune's classified instances is the same as tune's labels
	    // at instance's label, then increment a counter for accuracy
	    if (classify(tune.instances.get(i)).equals(tune.labels.get(convertLabel(tune.instances.get(i).label.toString()))))
		accuracyHits++;
	    
	}

	// Compute accuracy to return
	accuracy = accuracyHits / tune.instances.size();

	return accuracy;
    }

    /**
     * Traverse is a helper function to prune that traverses the Tree (T)
     * to be pruned
     */
    private List<DecTreeNode> traverse(DecTreeNode thisNode, 
			  List<DecTreeNode> prePrunedNodes) {

	// If this node is a terminal, end of this DFS recursive call
	if (thisNode.terminal)
	    return prePrunedNodes;
	// Otherwise add this node
	else
	    prePrunedNodes.add(thisNode);

	// Recursively (and DFS like) search the tree 
	for (int i = 0; i < thisNode.children.size(); i++) {

	    if (thisNode.children.size() != 0)
		return traverse(thisNode.children.get(i), prePrunedNodes);

	}

	return prePrunedNodes;
    }

    /**
     * From book page 702: DECISION-TREE-LEARNING
     * (examples, attributes, parent_examples) returns a tree
     *
     * I also looked at the slides provided by Collin. Decided to go
     * the route of passing in ROOT and return a child of type DecTreeNode
     * on each recursive call function return.
     */
    private DecTreeNode buildTree(DataSet train,
			     List<Instance> childInstance,
			     List<String> attributes,
			     List<Instance> parentInstance,
			     Integer parentAttribute) {

	/**
	 * "If examples is empty, then return 
	 * PLURALITY-VALUE(parent_examples)" 
	 * "The function PLURALITY-VALUE selects the most common output"
	 */
	// So, if "childInstance" is empty (or NULL but probably don't need
	// that, do a plurality value (most common) in a new child DecTreeNode
	if (childInstance.isEmpty() || childInstance == null)
	    return new DecTreeNode(pluralityValue(train, parentInstance), 
				   null, parentAttribute, true); 

	/**
	 * "...else if all examples have the same classification then return
	 * the classification"
	 */
	// So, if all "childInstance" have the same type or classification,
	// return any of the same classification in a new child DecTreeNode
	else if (isSameClassification(childInstance)) 
	    return new DecTreeNode(convertLabel
				   (train.labels.get
				    (childInstance.get(0).label)), 
				   null, parentAttribute, true);

	/**
	 * "...else if attributes is empty then return PLURALITY-VALUE(examples)"
	 */
	// So, if all attributes are empty, do a plurality value (most common)
	// and return a new child DecTreeNode
	else if (attributes.isEmpty())
	    return new DecTreeNode(pluralityValue(train, childInstance), 
				   null, parentAttribute, true);

	/**
	 * "...else"
	 * A <- argmax(e, attribute) IMPORTANCE(a examples)
	 * tree a new decision tree with root lest A
	 * for each value i of A do
	 *    era - {e E examples and e A = i}
	 *    subtree DECISION-TREE-LEARNING(attributes - A, examples)
	 *    add a branch to tree with label (A = i, and subtree subtree)
	 * return tree
	 *
	 * *****************OR in other words from ID3 Algo:
	 * Otherwise Begin
         *   A ← The Attribute that best classifies examples.
         *   Decision Tree attribute for Root = A.
         *   For each possible value, vi, of A,
         *     Add a new tree branch below Root, corresponding to the test 
	 *     A = vi.
         *     Let Examples(vi) be the subset of examples that have the value 
	 *     vi for A
         *     If Examples(vi) is empty
         *        Then below this new branch add a leaf node with label = mos 
	 *        common target value in the examples
         *     Else below this new branch add the subtree ID3 (Examples(vi), 
	 *     Target_Attribute, Attributes – {A})
         * End
	 */
	else {

	    int indexOfAttributeSel; // Need to find attribute selected
	    String strIndexOfAttributeSel; // String version of attribute

	    /**
             * Find attribute that best classifies examples (highest info
             * gain). Following algorithm, name of function is importance
             */
	    DecTreeNode A = importance(train, childInstance, attributes,
					  parentAttribute);

	    Integer attributeOfA = A.attribute;
	    Integer attributeIndexOfA = 0;
	    String attributeOfAStr;
	    DecTreeNode nextLevel;

	    // Need to obtain all "new" child attributes prior to recursion
	    List<String> newAttributes = new ArrayList<String>();

	    // Loop over all attributes in training data to find attribute of
	    // A
	    for (int i = 0; i < train.attributes.size(); i++) {

		// Collin said use .equals()
		if (convertAttribute(train.attributes.get(i)).equals(attributeOfA))
		    attributeIndexOfA = i;
	    
	    }

	    attributeOfAStr = train.attributes.get(attributeIndexOfA);

	    // Loop over attributes to get newAttributes based on 
	    // attributeIndexOfA
	    for (int i = 0; i < attributes.size(); i++) {
		
		if(i != attributeIndexOfA)
		    newAttributes.add(attributes.get(i));
	    
	    }
	    
	    // Now, just before recursion, we need the new childInstances
	    for (int i = 0; i < 
		     train.attributeValues.get(attributeOfAStr).size(); i++) {

		List<Instance> newChildInstance = new ArrayList<Instance>(); 

		// Add the "new" children to a list to pass into next call
		for (int j = 0; j < childInstance.size(); j++) {
		    if(childInstance.get(j).attributes.get(attributeIndexOfA).equals(i))
			newChildInstance.add(childInstance.get(j));
		}

		// The recursion to build the next level(s) of the tree
		nextLevel = buildTree(train, newChildInstance, 
				      newAttributes, 
				      childInstance, 
				      convertAttributeValue(attributeOfAStr, train.attributeValues.get(attributeOfAStr).get(i)));
		
		A.children.add(nextLevel);
	    }
	    
	    // Return the tree! yay done
	    return A;
	}

    }

    /**
     * pluralityValue, as given by the book algorithm, determines 
     * the most common label given a test instance
     */
    private Integer pluralityValue(DataSet train, 
				   List<Instance> instance) {
	
	int[] numberOfLabelHits = new int[train.labels.size()];
	int newMaxLabel = -1;
	int maxLabelToRet = -1;

	// For every label, increment counter how many times it appears
	// Will aide in finding most common label
	for (int i = 0; i < instance.size(); i++)
	    numberOfLabelHits[instance.get(i).label] += 1;

	// Now, need to find the most common label and return it
	for (int i = 0; i < numberOfLabelHits.length; i++) {

	    // This label has a higher frequency than previously found
	    if (numberOfLabelHits[i] > newMaxLabel) {
		
		// Update intermediate max label frequency and frequency to ret
		newMaxLabel = numberOfLabelHits[i];
		maxLabelToRet = i;

	    }
	}

	return convertLabel(train.labels.get(maxLabelToRet));
    }

    /**
     * isSameClassification, as given by the book algorithm, determines
     * if all instances have the same classification
     */
    private boolean isSameClassification(List<Instance> instance) {

	int thisLabel = -1; // Label to test against all other labels

	thisLabel = instance.get(0).label;

	// Loop through all instance labels
	for (int i = 1; i < instance.size(); i++) {
	    
	    // This label isn't of the type of label we're looking at, thus
	    // all labels not of same type
	    if (instance.get(i).label != thisLabel)
		return false;

	}

	// All labels are of the same classification
	return true;
    }

    /**
     * convertLabel is a simple helper function to retrieve integer form
     * of a label from its string form.
     */
    private Integer convertLabel(String label) {

	for (int i = 0; i < this.labels.size(); i++) {
	    
	    if (label.equals(this.labels.get(i)))
		return i;
	
	}
	
	// Shouldn't reach here but compiler yelling
        return 0;
    }


    /**
     * convertAttribute is a simple helper function to retrieve integer form
     * of an attribute from its string form.
     */
    private Integer convertAttribute(String attribute) {

        for (int i = 0; i < this.attributes.size(); i++) {

	    if (attribute.equals(this.attributes.get(i)))
		return i;
	    
	}

	// Shouldn't reach here but compiler yelling
        return 0;
    }

    /**
     * convertAttributeValue is a simple helper function to retrieve 
     * integer form of an attributeValue from its string form.
     */
    private Integer convertAttributeValue(String attribute, 
					  String discreteValue) {

        for (int i = 0; i < attributeValues.get(attribute).size(); i++) {

	    if (discreteValue.equals(attributeValues.get(attribute).get(i)))
		return i;

	}

	// Shouldn't reach here but compiler was yelling
	return 0;
    }


    /**
     * importance is a helper function to buildTree that selects the
     * highest information gain among all attributes and returns a
     * new child node with that max IG.
     */
    private DecTreeNode importance(DataSet train, 
				   List<Instance> instance,
				   List<String> attributes,
				   Integer parentAttribute) {
	
	int maxAttribute = -1; // Attribute with max IG
	int maxLabel = -1; // Absolute max IG
	int thisAttributeIndex = -1; // Attribute we're looking at
	int thisMaxLabel = -1; // Intermediate max IG
	int maxLabelHits = -1; // Max label hits with max IG

	double HS = 0.0; // Entropy
	double HYX = 0.0; // Conditional Entropy
	double IG = 0.0; // Information gain

	double maxIG = 0.0; // Maximum information gain among all attributes

	// Figure out entropy for IG calc for max IG
	HS = entropy(train, instance);

	// Loop through all attributes
	for (int i = 0; i < attributes.size(); i++) {
    
	    for (int j = 0; j < train.attributes.size(); j++) {

		// Collin said to use .equals()
		if (train.attributes.get(j).equals(attributes.get(i)))
		    thisAttributeIndex = j;
	    }

	    // Figure out conditional entropy for IG calc for max IG
	    HYX = conditionalEntropy(train, instance, attributes,
				     attributes.get(i),
				     thisAttributeIndex);
	    
	    // This IG
	    IG = HS - HYX;

	    // Need to find training info with most amount of labels
	    for(int j = 0; j < train.labels.size(); j++){
		
		if(convertLabel(labels.get(j)) > maxLabelHits){
		    maxLabelHits = convertLabel(labels.get(j));
		    thisMaxLabel = j;
		}
	    
	    }

	    // Obtain training data with most IG
	    if (IG > maxIG) {

		// Swap out new max IG since maxIG < IG
		maxAttribute = convertAttribute(attributes.get(i));
		maxIG = IG;
		maxLabel = thisMaxLabel;

	    }
	}

	// Return a new child node with maximum information gain
	return new DecTreeNode(convertLabel(train.labels.get(maxLabel)), 
			       maxAttribute, parentAttribute, false);
    
    }

    @Override
    public String classify(Instance instance) {
	
	// TODO: add code here
	
	/**
	 * "The public String classify(Instance instance) takes an
	 * instance as its input and gives the classification output
	 * of the built decision tree as a string. You do not need
	 * to worry about printing. That part is already handled
	 * within the code."
	 */

	// Recursive call helper
	String classify = classifyRecursion(instance, root);

	return classify;
    }

    /**
     * Needed function classifyHelper to do recursive calls
     * through subtrees (children)
     */
    public String classifyRecursion(Instance instance,
				    DecTreeNode thisNode) {
	
	String classify = null;

	// If end of traverse, roll back
	if (thisNode.terminal)
	    return thisNode.label.toString();
	
	/**
	 * Now, recursively find the classification by calling classifyHelper
	 * which compares this child's parent's value to the 
	 * instance's attribute's node attribute.
	 */
	for (int i = 0; i < thisNode.children.size(); i++) {

	    if(thisNode.children.get(i).parentAttributeValue.equals(instance.attributes.get(thisNode.attribute)))
		classify = classifyRecursion(instance, 
					     thisNode.children.get(i));

	} 
	
	return classify; 
    }
    

    @Override
    /**
     * Print the decision tree in the specified format
     */
	public void print() {
	
	printTreeNode(root, null, 0);
    }

    /**
     * Prints the subtree of the node
     * with each line prefixed by 4 * k spaces.
     */
    public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
	StringBuilder sb = new StringBuilder();
	for (int i = 0; i < k; i++) {
	    sb.append("    ");
	}
	String value;
	if (parent == null) {
	    value = "ROOT";
	} else{
	    String parentAttribute = attributes.get(parent.attribute);
	    value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
	}
	sb.append(value);
	if (p.terminal) {
	    sb.append(" (" + labels.get(p.label) + ")");
	    System.out.println(sb.toString());
	} else {
	    sb.append(" {" + attributes.get(p.attribute) + "?}");
	    System.out.println(sb.toString());
	    for(DecTreeNode child: p.children) {
		printTreeNode(child, p, k+1);
	    }
	}
    }

    @Override
    public void rootInfoGain(DataSet train) {

	this.labels = train.labels;
	this.attributes = train.attributes;
	this.attributeValues = train.attributeValues;
	// TODO: add code here
	
	/**
	 * From Wikipedia: "In general terms, the expected information
	 * gain is the change in information entropy H from a prior
	 * state to a state that takes some information as given:
	 * IG(T, a) = H(T) - H(T|a)".
	 *
	 * "Information gain IG(A) is the measure of the difference in
	 * entropy from before to after the set S in split on atribute A.
	 * In other words, how much uncertainty in S was reduced after 
	 * splitting set S on attribute A."
	 *
	 * IG(A, S) = H(S) - summation of [p(t) * H(t)] over t within T
	 */
	double HTEntropy = 0.0; // Normal (non-conditional entropy)
	double IG = 0.0; // Information gain 
	double HTConditionalEntropy = 0.0; // Conditional entropy
	int thisIndex = -1; // Index where training attribute matches 

	// Compute entropy
	HTEntropy = entropy(train, 
			    train.instances);

	// For each attribute "a", compute conditional entropy, print
	for (int i = 0; i < train.attributes.size(); i++) {

	    // Find attribute that matches training attribute
	    for (int j = 0; j < train.attributes.size(); j++) {

		// Collin said use .equals()
                if (train.attributes.get(j).equals(train.attributes.get(i)))
                    thisIndex = j;

            }
	    
	    // Calculate conditional entropy
	    HTConditionalEntropy = conditionalEntropy(train,
						      train.instances,
						      attributes, 
						      train.attributes.get(i),
						      thisIndex);

	    // Calculate information gain
	    IG = HTEntropy - HTConditionalEntropy;
	    
	    // Now, print in that nice decimal format
	    System.out.printf("%s %.5f\n", train.attributes.get(thisIndex) + " ", IG);
	}
    }

    /**
     * From Wikipedia:
     * "Entropy H(S) is a measure of the amount of
     * uncertainty in the (data) set S (i.e. entropy
     * characterizes the (data) set S).
     * In ID3, entropy is calculated for each remaining attribute.
     *
     * H(S) = - summation over[p(x) * logbase2(p(x))] of x within X
     *
     * When H(S) = 0, the set S is perfectly classified (i.e. all elements
     * in S are of the same class)."
     *
     * I.e. for random variable X with probability P(x), the entropy
     * is the average (or expected) amount of information obtained
     * by observing x
     */
    private double entropy(DataSet train, List<Instance> instance) {
	
	double Px = 0.0; // Per algorithm
	double log2Px = 0.0; // Per algorithm
	
	// Need to find out how many times each label occurs per attribute
	double [] numberOfLabelsPerAttribute = 
	    new double[train.labels.size()];

	double HSEntropy = 0.0; // The entropy to return

        // Calculate frequency labels occur to calculate P(x)
        for (int i = 0; i < instance.size(); i++)
            numberOfLabelsPerAttribute[instance.get(i).label] += 1;

        // Compute entropy on labels per attribute
        for (int i = 0; i < numberOfLabelsPerAttribute.length; i++) {

	    // Calculate Px
	    Px = numberOfLabelsPerAttribute[i] / instance.size();
		
	    // Calculate log2(Px). I don't think there is log base 2
	    // so using logarithm rules (log10() / log2())
	    log2Px = Math.log(numberOfLabelsPerAttribute[i] /
			      instance.size()) / Math.log(2.0);
		
	    // Negative summation
	    HSEntropy +=  -(Px * log2Px);

        }

        return HSEntropy;
    }

    /**
     * From the CS 540 slides provided by Collin
     * (and I looked at Chuck Dyer's):
     * "Conditional Entropy: H(Y|X) is the remaining entropy of Y given
     * X OR the expected (or average) entropy of P(y|x)
     * H(Y|X) = - summation over x of (P(x)) * summation over y
     * of H(Y|X) = x
     * OR
     * H(Y|X) = -summation over x of (P(x)) * summation over y
     * of P(y|x) logbase2(P(y|x))"
     *
     * "Weighted sum of the entropy of each subset of the examples
     * partitioned by the possible values of the attribute X"
     *
     * 0 <= H(Y | X) <= 1
     */
    private double conditionalEntropy(DataSet train, List<Instance> instance,
                                      List<String> attributes, 
				      String thisAttribute, int currIndex) {
	
	// Entropy parts of equation
        double Px = 0.0; // Per algorithm
        double log2Px = 0.0; // Per algorithm

        // Conditional Entropy parts of equation
        double Pr = 0.0;
	
	// Allocate enough space for HYX per attribute
	double [] HYX = 
	    new double[train.attributeValues.get(thisAttribute).size()];
        
	// Number of attribute hits per attribute
	double [] numberOfAttributeHits = 
	    new double[train.attributeValues.get(thisAttribute).size()];

	// Number of label hits per label per attribute
        double [][] numberOfLabelHits = 
	    new double[numberOfAttributeHits.length][train.labels.size()];

	double HYXRet = 0.0; // Conditional Entropy to return

	// Obtain number of attribute hits per attribute and number of label
	// hits per label per attribute
        for (int i = 0; i < instance.size(); i++) {

	    numberOfLabelHits
		[instance.get(i).attributes.get(currIndex)]
		[instance.get(i).label] += 1;
	    numberOfAttributeHits
		[instance.get(i).attributes.get(currIndex)] += 1;

        }

	// For all attributes in training set:
	for (int i = 0; i < train.attributeValues.get(thisAttribute).size(); 
	     i++) {

	    // For all labels in training set:
	    for (int j = 0; j < train.labels.size(); j++) {

		// Don't add this label of this attribute as it can mess
		// with conditional entropy calculations
		if (numberOfLabelHits[i][j] == 0 || 
		    numberOfAttributeHits[i] == 0)
		     continue;

		// Similar to entropy, calculate Px, log2Px for conditional
		// entropy
		Px = numberOfLabelHits[i][j] / numberOfAttributeHits[i];
		log2Px = Math.log(numberOfLabelHits[i][j] / 
				  numberOfAttributeHits[i]) / Math.log(2.0);
                
		// Entropy part of calculation is done (yay)
		HYX[i] += -(Px * log2Px);
            }
	    
	    // Conditional entropy part: conditional probability
	    Pr = numberOfAttributeHits[i] / train.instances.size();

	    // Multiply entropy by probablity to get conditional entropy
            HYX[i] = HYX[i] * Pr;
	}


	// Perform the summation
	for (int i = 0; i < HYX.length; i++)
	    HYXRet += HYX[i];
	
        return HYXRet;
    }

}
