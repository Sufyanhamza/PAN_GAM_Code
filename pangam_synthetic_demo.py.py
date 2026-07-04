import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch
from statsmodels.stats.multitest import multipletests
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Tuple, Union, Optional
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import sys

# =============================================================================
# Configuration & Logging Setup
# =============================================================================

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Pan-GAM] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class PanGAMConfig:
    """
    Configuration object for the Pan-GAM pipeline.
    
    Attributes:
        jaccard_threshold (float): Threshold for clustering CTUs (epsilon = 0.05). [cite: 101]
        min_group_size (int): Minimum size for a phenotypic group to be retained. [cite: 93]
        min_control_size (int): Minimum size for the specific susceptible control group (Omega_S).
        alpha (float): Significance threshold for FDR control. [cite: 133]
        ppv_threshold (float): Minimum Positive Predictive Value to accept a hit. [cite: 134]
        n_jobs (int): Number of parallel threads for distance calculation.
    """
    jaccard_threshold: float = 0.05
    min_group_size: int = 2
    min_control_size: int = 10  # Added based on our discussion
    alpha: float = 0.05
    ppv_threshold: float = 0.90
    n_jobs: int = 4

# =============================================================================
# Module 1: Phenotypic Partitioner
# =============================================================================

class PhenotypicPartitioner:
    """
    Implements the Phenotypic Partitioning stage of Pan-GAM.
    
    Mathematically partitions the set of isolates S into K disjoint subsets 
    based on unique phenotypic vectors y_i. [cite: 89-91]
    """
    
    def __init__(self, min_group_size: int = 2):
        self.min_group_size = min_group_size
        self.groups: Dict[tuple, List[str]] = {}
        self.isolate_to_group: Dict[str, tuple] = {}

    def fit(self, phenotype_matrix: pd.DataFrame) -> 'PhenotypicPartitioner':
        """
        Stratifies isolates into discrete groups.
        
        Args:
            phenotype_matrix: Binary DataFrame (Isolates x Drugs). 1=Resistant, 0=Susceptible.
        """
        logger.info("Stage 1: Starting Phenotypic Partitioning...")
        
        # Group by all columns to find unique resistance signatures
        # y_i represents the resistance profile vector [cite: 70]
        grouped = phenotype_matrix.groupby(list(phenotype_matrix.columns))
        
        total_groups = 0
        retained_groups = 0
        
        for signature, group_df in grouped:
            total_groups += 1
            isolate_ids = group_df.index.tolist()
            
            # Filter groups with |Ck| < 2 [cite: 93]
            if len(isolate_ids) >= self.min_group_size:
                self.groups[signature] = isolate_ids
                for iso in isolate_ids:
                    self.isolate_to_group[iso] = signature
                retained_groups += 1
        
        logger.info(f"Partitioning complete. Total unique profiles: {total_groups}. "
                    f"Retained groups (|Ck| >= {self.min_group_size}): {retained_groups}.")
        return self

    def get_groups(self) -> Dict[tuple, List[str]]:
        return self.groups

# =============================================================================
# Module 2: Collinearity-Penalized Clustering (The HGT Module)
# =============================================================================

class HGTClusteringModule:
    """
    Implements the Collinearity-Penalized Clustering module.
    
    Resolves the 'hitchhiker effect' by aggregating linked genes into 
    Co-transfer Units (CTUs) based on Jaccard distance. [cite: 97-101]
    """
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.ctu_map: Dict[int, List[str]] = {} # CTU_ID -> List of Gene Names
        self.gene_to_ctu: Dict[str, int] = {}
        self.representative_matrix: Optional[pd.DataFrame] = None

    def fit_transform(self, gene_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Performs hierarchical clustering on the feature space and collapses features.
        
        Args:
            gene_matrix: Binary DataFrame (Isolates x Genes). G_ij = 1 if present. [cite: 80]
            
        Returns:
            Reduced DataFrame (Isolates x CTUs).
        """
        logger.info("Stage 2: Starting Collinearity-Penalized Clustering (HGT Module)...")
        n_isolates, n_genes = gene_matrix.shape
        logger.info(f"Input feature space: {n_genes} genes across {n_isolates} isolates.")

        # Transpose to cluster genes (features), not isolates
        X = gene_matrix.T.values # Shape: (M genes, N isolates)
        gene_names = gene_matrix.columns.tolist()

        # 1. Calculate Pairwise Jaccard Distance [cite: 99]
        # d_J(g_j, g_l) = 1 - (|Intersection| / |Union|)
        # Note: 'pdist' with metric='jaccard' computes this efficiently.
        logger.info("Calculating pairwise Jaccard distances...")
        try:
            # For very large matrices, this should be chunked. 
            # Here we assume it fits in memory for the "complex" demo.
            dist_matrix = distance.pdist(X, metric='jaccard')
        except MemoryError:
            logger.error("Gene matrix too large for in-memory dense distance calculation.")
            raise

        # 2. Hierarchical Clustering [cite: 100]
        # Using average linkage (UPGMA) or complete linkage as is standard in constructing CTUs.
        logger.info("Performing hierarchical clustering...")
        linkage_matrix = sch.linkage(dist_matrix, method='average')

        # 3. Form Clusters (CTUs) based on threshold epsilon [cite: 101]
        cluster_labels = sch.fcluster(linkage_matrix, t=self.threshold, criterion='distance')
        
        # 4. Collapse Feature Space [cite: 102]
        logger.info("Collapsing features into Co-transfer Units (CTUs)...")
        ctu_dict = {}
        
        # Group genes by cluster label
        df_transposed = gene_matrix.T
        df_transposed['cluster'] = cluster_labels
        
        # For the representative vector, we take the consensus or the "driver" (centroid).
        # In binary matrices with distance < 0.05, they are almost identical.
        # We take the first gene's pattern as the representative for the CTU.
        ctu_representatives = {}
        
        for cluster_id, sub_df in df_transposed.groupby('cluster'):
            genes_in_ctu = sub_df.index.tolist()
            
            # Generate a CTU ID
            ctu_id = f"CTU_{cluster_id:05d}"
            self.ctu_map[ctu_id] = genes_in_ctu
            for gene in genes_in_ctu:
                self.gene_to_ctu[gene] = ctu_id
            
            # Extract representative vector (from the original matrix)
            # We assume perfect collinearity for the purpose of the model [cite: 103]
            rep_gene = genes_in_ctu[0]
            ctu_representatives[ctu_id] = gene_matrix[rep_gene]

        # Create the reduced matrix G' [cite: 102]
        self.representative_matrix = pd.DataFrame(ctu_representatives)
        
        reduction_pct = (1 - (self.representative_matrix.shape[1] / n_genes)) * 100
        logger.info(f"Feature space reduced to {self.representative_matrix.shape[1]} CTUs.")
        logger.info(f"Dimensionality reduction: {reduction_pct:.2f}% [cite: 118]")
        
        return self.representative_matrix

# =============================================================================
# Module 3: Statistical Association Engine
# =============================================================================

class HypergeometricEngine:
    """
    Performs the statistical core of Pan-GAM using complex control groups.
    """
    
    def __init__(self, config: PanGAMConfig):
        self.config = config

    def run_association(self, 
                       ctu_matrix: pd.DataFrame, 
                       pheno_matrix: pd.DataFrame, 
                       groups: Dict[tuple, List[str]],
                       isolate_group_map: Dict[str, tuple]) -> pd.DataFrame:
        """
        Executes association testing for all drugs.
        """
        logger.info("Stage 3: Running Hypergeometric Association Testing...")
        
        results = []
        drugs = pheno_matrix.columns.tolist()
        
        for drug_idx, target_drug in enumerate(drugs):
            logger.info(f"Processing target drug: {target_drug}")
            
            # 1. Define Populations Omega_R and Omega_S [cite: 123-124]
            # Omega_R: Isolates in groups resistant to target drug
            # Omega_S: Isolates in groups susceptible to target drug BUT resistant to others
            
            omega_r_isolates = []
            omega_s_isolates = []
            
            for signature, isolates in groups.items():
                # signature is a tuple of 0s and 1s corresponding to drugs
                is_resistant_to_target = (signature[drug_idx] == 1)
                is_resistant_to_any_other = (sum(signature) - signature[drug_idx]) > 0
                
                if is_resistant_to_target:
                    omega_r_isolates.extend(isolates)
                elif is_resistant_to_any_other:
                    # Ideally, we use complex susceptible.
                    omega_s_isolates.extend(isolates)
                else:
                    # Pan-susceptible (C0). 
                    # Note: The manuscript implies using Omega_S primarily to cancel noise.
                    # We utilize the fallback logic discussed.
                    pass

            # Fallback Logic for Statistical Power check
            if len(omega_s_isolates) < self.config.min_control_size:
                logger.warning(f"Omega_S size ({len(omega_s_isolates)}) < threshold. Including Pan-Susceptible isolates.")
                # Add pan-susceptible to control group to boost power
                for signature, isolates in groups.items():
                    if sum(signature) == 0:
                        omega_s_isolates.extend(isolates)

            n_resistant = len(omega_r_isolates)
            n_susceptible = len(omega_s_isolates)
            
            if n_resistant == 0 or n_susceptible == 0:
                logger.warning(f"Skipping {target_drug}: Insufficient cases/controls.")
                continue
                
            # Pre-fetch CTU data for these populations
            try:
                ctu_r = ctu_matrix.loc[omega_r_isolates]
                ctu_s = ctu_matrix.loc[omega_s_isolates]
            except KeyError:
                logger.error("Isolate ID mismatch between phenotype and genome matrices.")
                continue

            # 2. Iterate over every CTU to calculate Contingency Table [cite: 125]
            # Table:
            #       | Present (1) | Absent (0) | Total
            # Res   | a           | b          | n_resistant
            # Sus   | c           | d_count    | n_susceptible
            # Total | a+c         | b+d        | N
            
            # Vectorized calculation for speed
            a_vector = ctu_r.sum(axis=0) # Sum of presence in Resistant
            c_vector = ctu_s.sum(axis=0) # Sum of presence in Susceptible
            
            # Calculate Hypergeometric P-values
            # The survival function (sf) is 1 - CDF, equivalent to the test for enrichment.
            # M = population size (N)
            # n = total number of successes in population (a + c)
            # N = sample size drawn (n_resistant)
            # k = successes in sample (a)
            
            M = n_resistant + n_susceptible
            n_total_presence = a_vector + c_vector
            N_draw = n_resistant
            
            p_values = stats.hypergeom.sf(a_vector - 1, M, n_total_presence, N_draw)
            
            # Calculate Odds Ratios
            # OR = (a*d) / (b*c)
            # Smooth with +0.5 to avoid division by zero (Haldane correction)
            a = a_vector
            b = n_resistant - a
            c = c_vector
            d = n_susceptible - c
            
            odds_ratios = (a * d) / (b * c.replace(0, 1e-9)) # basic avoidance
            # Better implementation with Haldane:
            odds_ratios = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            
            # Store results
            temp_res = pd.DataFrame({
                'Drug': target_drug,
                'CTU': ctu_matrix.columns,
                'a': a, 'b': b, 'c': c, 'd': d,
                'P_value': p_values,
                'Odds_Ratio': odds_ratios,
                'PPV': a / (a + c + 1e-9) # Positive Predictive Value
            })
            results.append(temp_res)

        if not results:
            return pd.DataFrame()

        full_results = pd.concat(results, ignore_index=True)
        
        # 3. False Discovery Rate Control [cite: 130-133]
        logger.info("Stage 4: Applying Benjamini-Hochberg FDR correction...")
        
        # Apply BH correction per drug
        full_results['P_adj'] = 1.0
        for drug in drugs:
            mask = full_results['Drug'] == drug
            if mask.sum() > 0:
                p_vals = full_results.loc[mask, 'P_value']
                _, p_adj, _, _ = multipletests(p_vals, alpha=self.config.alpha, method='fdr_bh')
                full_results.loc[mask, 'P_adj'] = p_adj

        return full_results

    def filter_significant(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Filters for significant hits based on P_adj and PPV. [cite: 134]
        """
        mask = (results['P_adj'] <= self.config.alpha) & \
               (results['PPV'] >= self.config.ppv_threshold)
        return results[mask].sort_values('P_adj')

# =============================================================================
# Main Pipeline Class
# =============================================================================

class PanGAM(BaseEstimator):
    """
    The main interface for the Pan-GAM framework.
    Wrapper class integrating Partitioning, Clustering, and Testing.
    """
    
    def __init__(self, 
                 jaccard_threshold=0.05, 
                 min_group_size=2, 
                 n_jobs=1):
        self.config = PanGAMConfig(
            jaccard_threshold=jaccard_threshold,
            min_group_size=min_group_size,
            n_jobs=n_jobs
        )
        self.partitioner = PhenotypicPartitioner(min_group_size)
        self.hgt_module = HGTClusteringModule(jaccard_threshold)
        self.engine = HypergeometricEngine(self.config)
        
        self.results_ = None
        self.ctu_map_ = None
        
    def fit(self, gene_matrix: pd.DataFrame, pheno_matrix: pd.DataFrame):
        """
        Run the full Pan-GAM pipeline.
        
        Args:
            gene_matrix: Binary (N isolates x M genes)
            pheno_matrix: Binary (N isolates x D drugs)
        """
        # Validate Inputs
        if not gene_matrix.index.equals(pheno_matrix.index):
            raise ValueError("Index mismatch between Gene and Phenotype matrices.")
            
        # Step 1: Phenotypic Partitioning
        self.partitioner.fit(pheno_matrix)
        groups = self.partitioner.get_groups()
        isolate_map = self.partitioner.isolate_to_group
        
        # Step 2: HGT Module (CTU construction)
        # This reduces M genes to M' CTUs
        ctu_matrix = self.hgt_module.fit_transform(gene_matrix)
        self.ctu_map_ = self.hgt_module.ctu_map
        
        # Step 3: Association Testing
        raw_results = self.engine.run_association(
            ctu_matrix=ctu_matrix,
            pheno_matrix=pheno_matrix,
            groups=groups,
            isolate_group_map=isolate_map
        )
        
        # Step 4: Filtering
        self.results_ = self.engine.filter_significant(raw_results)
        
        logger.info(f"Pan-GAM pipeline complete. Found {len(self.results_)} significant associations.")
        return self

# =============================================================================
# Execution Simulation (Main)
# =============================================================================

if __name__ == "__main__":
    # Simulate a small run to demonstrate functionality
    try:
        logger.info("Initializing Pan-GAM Synthetic Run...")
        
        # 1. Create Synthetic Data (representing S. aureus)
        np.random.seed(42)
        n_iso = 200
        n_genes = 500
        n_drugs = 3
        
        # Isolates index
        isolates = [f"Iso_{i}" for i in range(n_iso)]
        
        # Phenotypes (Drugs: Methicillin, Erythromycin, Gentamicin)
        # Create some correlation structure
        pheno_data = np.random.choice([0, 1], size=(n_iso, n_drugs), p=[0.7, 0.3])
        pheno_df = pd.DataFrame(pheno_data, index=isolates, columns=['Methicillin', 'Erythromycin', 'Gentamicin'])
        
        # Genes (G Matrix)
        # Create a "driver" gene that perfectly matches Methicillin (simulating mecA)
        gene_data = np.random.choice([0, 1], size=(n_iso, n_genes), p=[0.9, 0.1])
        gene_df = pd.DataFrame(gene_data, index=isolates, columns=[f"gene_{i}" for i in range(n_genes)])
        
        # Inject Signal: mecA (gene_0) and its hitchhikers (gene_1, gene_2)
        # They should be perfectly collinear and match Methicillin phenotype
        target_indices = pheno_df[pheno_df['Methicillin'] == 1].index
        gene_df.loc[target_indices, 'gene_0'] = 1 # mecA
        gene_df.loc[target_indices, 'gene_1'] = 1 # mecR1 (Hitchhiker)
        gene_df.loc[target_indices, 'gene_2'] = 1 # mecI (Hitchhiker)
        
        # Inject Noise: Cross-resistance artifact
        # Gene_50 is linked to Erythromycin, but we want to see if it shows up in Methicillin
        erythro_indices = pheno_df[pheno_df['Erythromycin'] == 1].index
        gene_df.loc[erythro_indices, 'gene_50'] = 1 
        
        # 2. Instantiate and Run Pan-GAM
        model = PanGAM(jaccard_threshold=0.05, min_group_size=2)
        model.fit(gene_df, pheno_df)
        
        # 3. Inspect Results
        print("\n=== TOP HITS ===")
        print(model.results_.head(10))
        
        # Verify CTU structure (should see gene_0, gene_1, gene_2 grouped)
        print("\n=== CTU Structure Validation ===")
        for idx, row in model.results_.head(1).iterrows():
            ctu_id = row['CTU']
            genes = model.ctu_map_[ctu_id]
            print(f"Top Hit {ctu_id} contains genes: {genes}")
            
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise
