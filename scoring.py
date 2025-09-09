import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List

class FIPeerScoring:
    """
    Fixed Income Fund Peer Scoring System
    Calculates similarity scores between funds within the same Morningstar category
    """
    
    def __init__(self, 
                 currency_weight: float = 30,
                 passive_weight: float = 10,
                 fee_weight: float = 25,
                 region_weight: float = 20,
                 sector_weight: float = 15):
        """
        Initialize scoring weights (should sum to 100)
        
        Parameters:
        -----------
        currency_weight: Weight for currency matching (default: 30%)
        passive_weight: Weight for active/passive matching (default: 10%)
        fee_weight: Weight for fee band similarity (default: 25%)
        region_weight: Weight for region matching (default: 20%)
        sector_weight: Weight for primary sector matching (default: 15%)
        """
        self.weights = {
            'currency': currency_weight,
            'passive': passive_weight,
            'fee': fee_weight,
            'region': region_weight,
            'sector': sector_weight
        }
        
        # Validate weights sum to 100
        total_weight = sum(self.weights.values())
        if abs(total_weight - 100) > 0.01:
            print(f"Warning: Weights sum to {total_weight}, not 100. Normalizing...")
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total_weight) * 100
    
    def currency_score(self, fund1_currency: str, fund2_currency: str) -> float:
        """
        Calculate currency matching score
        
        Returns:
        --------
        100 if currencies match, 0 if different, 25 if either is NaN
        """
        if pd.isna(fund1_currency) or pd.isna(fund2_currency):
            return 25  # Partial score for missing data
        
        return 100 if fund1_currency == fund2_currency else 0
    
    def passive_score(self, fund1_passive: bool, fund2_passive: bool) -> float:
        """
        Calculate active/passive matching score
        
        Returns:
        --------
        100 if both active or both passive, 0 if different, 50 if either is NaN
        """
        if pd.isna(fund1_passive) or pd.isna(fund2_passive):
            return 50  # Neutral score for missing data
        
        return 100 if fund1_passive == fund2_passive else 0
    
    def fee_band_score(self, fund1_band: int, fund2_band: int) -> float:
        """
        Calculate fee band similarity score
        
        Fee bands assumed to be 1-5 where:
        Band 1: Lowest fees (within 1 std dev below mean)
        Band 2: Low fees (between -1 and 0 std dev)
        Band 3: Average fees (between 0 and +1 std dev)
        Band 4: High fees (between +1 and +2 std dev)
        Band 5: Highest fees (above +2 std dev)
        
        Returns:
        --------
        Score based on band difference (0-100)
        """
        if pd.isna(fund1_band) or pd.isna(fund2_band):
            return 30  # Lower score for missing data
        
        band_diff = abs(fund1_band - fund2_band)
        
        # Score mapping based on band difference
        score_map = {
            0: 100,  # Same band
            1: 75,   # Adjacent bands
            2: 50,   # 2 bands apart
            3: 25,   # 3 bands apart
            4: 0     # 4 bands apart
        }
        
        return score_map.get(band_diff, 0)
    
    def region_score(self, fund1_region: str, fund2_region: str) -> float:
        """
        Calculate region matching score
        
        Regions: 'emerging', 'developed', 'other'
        
        Returns:
        --------
        100 if regions match, partial score for related regions, 0 if completely different
        """
        if pd.isna(fund1_region) or pd.isna(fund2_region):
            return 30  # Lower score for missing data
        
        fund1_region = str(fund1_region).lower()
        fund2_region = str(fund2_region).lower()
        
        if fund1_region == fund2_region:
            return 100
        
        # Partial scores for somewhat related regions
        region_similarity = {
            ('emerging', 'other'): 40,
            ('other', 'emerging'): 40,
            ('developed', 'other'): 40,
            ('other', 'developed'): 40,
            ('emerging', 'developed'): 20,
            ('developed', 'emerging'): 20
        }
        
        return region_similarity.get((fund1_region, fund2_region), 0)
    
    def sector_score(self, fund1_sector: str, fund2_sector: str) -> float:
        """
        Calculate primary sector matching score
        
        Returns:
        --------
        100 if sectors match exactly, partial score for related sectors, 0 if different
        """
        if pd.isna(fund1_sector) or pd.isna(fund2_sector):
            return 30  # Lower score for missing data
        
        fund1_sector = str(fund1_sector).lower()
        fund2_sector = str(fund2_sector).lower()
        
        if fund1_sector == fund2_sector:
            return 100
        
        # Define related sectors for partial scoring (customize based on your sector definitions)
        related_sectors = {
            'government': ['sovereign', 'municipal', 'agency'],
            'corporate': ['investment grade', 'high yield', 'credit'],
            'securitized': ['mbs', 'abs', 'cmbs'],
            'mixed': ['multi-sector', 'diversified', 'aggregate']
        }
        
        # Check if sectors are related
        for main_sector, related_list in related_sectors.items():
            if main_sector in fund1_sector or main_sector in fund2_sector:
                for related in related_list:
                    if related in fund1_sector or related in fund2_sector:
                        return 60  # Partial score for related sectors
        
        return 0
    
    def calculate_peer_score(self, fund1: Dict, fund2: Dict) -> Tuple[float, Dict]:
        """
        Calculate overall peer similarity score between two funds
        
        Parameters:
        -----------
        fund1, fund2: Dictionaries containing fund attributes:
            - 'currency': Currency code
            - 'is_passive': Boolean indicating if fund is passive
            - 'fee_band': Integer (1-5) indicating fee band
            - 'region': String ('emerging', 'developed', 'other')
            - 'primary_sector': String indicating primary sector
        
        Returns:
        --------
        Tuple of (overall_score, component_scores_dict)
        """
        # Calculate individual component scores
        scores = {
            'currency': self.currency_score(
                fund1.get('currency'), 
                fund2.get('currency')
            ),
            'passive': self.passive_score(
                fund1.get('is_passive'), 
                fund2.get('is_passive')
            ),
            'fee': self.fee_band_score(
                fund1.get('fee_band'), 
                fund2.get('fee_band')
            ),
            'region': self.region_score(
                fund1.get('region'), 
                fund2.get('region')
            ),
            'sector': self.sector_score(
                fund1.get('primary_sector'), 
                fund2.get('primary_sector')
            )
        }
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[component] * self.weights[component] / 100 
            for component in scores
        )
        
        return overall_score, scores
    
    def score_peers_for_fund(self, 
                            target_fund_id: str,
                            all_funds_df: pd.DataFrame,
                            exclude_passive: bool = True) -> pd.DataFrame:
        """
        Score all funds in the same M* category against a target fund
        
        Parameters:
        -----------
        target_fund_id: ID of the target fund
        all_funds_df: DataFrame with all fund data including M* category
        exclude_passive: If True, exclude passive funds from peer group (default: True)
        
        Returns:
        --------
        DataFrame with peer scores for funds in the same M* category
        """
        # Get target fund data
        target_fund_data = all_funds_df[all_funds_df['fund_id'] == target_fund_id]
        
        if target_fund_data.empty:
            raise ValueError(f"Fund {target_fund_id} not found in dataset")
        
        target_fund = target_fund_data.iloc[0].to_dict()
        target_category = target_fund.get('morningstar_category')
        
        if pd.isna(target_category):
            print(f"Warning: Fund {target_fund_id} has no M* category")
            return pd.DataFrame()
        
        # Get all funds in the same M* category (excluding the target fund itself)
        same_category_funds = all_funds_df[
            (all_funds_df['morningstar_category'] == target_category) & 
            (all_funds_df['fund_id'] != target_fund_id)
        ].copy()
        
        # Optionally exclude passive funds
        if exclude_passive and not target_fund.get('is_passive', False):
            same_category_funds = same_category_funds[
                same_category_funds['is_passive'] != True
            ]
        
        if same_category_funds.empty:
            print(f"No comparable funds found in category {target_category}")
            return pd.DataFrame()
        
        # Calculate scores for each potential peer
        results = []
        for idx, row in same_category_funds.iterrows():
            candidate = row.to_dict()
            overall_score, component_scores = self.calculate_peer_score(target_fund, candidate)
            
            result = {
                'fund_id': row['fund_id'],
                'fund_name': row.get('fund_name', f'Fund_{idx}'),
                'morningstar_category': target_category,
                'peer_score': round(overall_score, 2),
                'currency': row.get('currency'),
                'is_passive': row.get('is_passive'),
                'fee_band': row.get('fee_band'),
                'region': row.get('region'),
                'primary_sector': row.get('primary_sector'),
                **{f'{k}_score': round(v, 2) for k, v in component_scores.items()}
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('peer_score', ascending=False)
        
        return results_df
    
    def create_peer_groups_for_firm_funds(self,
                                         firm_fund_ids: List[str],
                                         all_funds_df: pd.DataFrame,
                                         min_score: float = 70,
                                         max_peers: int = 30,
                                         exclude_passive: bool = True,
                                         output_summary: bool = True) -> Dict:
        """
        Create peer groups for specified firm funds based on M* categories
        
        Parameters:
        -----------
        firm_fund_ids: List of fund IDs belonging to your firm
        all_funds_df: DataFrame with all fund data including M* category
        min_score: Minimum score to be considered a peer (default: 70)
        max_peers: Maximum number of peers per fund (default: 30)
        exclude_passive: If True, exclude passive funds from peer groups (default: True)
        output_summary: If True, print summary statistics (default: True)
        
        Returns:
        --------
        Dictionary with detailed peer group information for each firm fund
        """
        peer_groups = {}
        summary_stats = []
        
        for fund_id in firm_fund_ids:
            # Get fund info
            fund_info = all_funds_df[all_funds_df['fund_id'] == fund_id]
            
            if fund_info.empty:
                print(f"Warning: Fund {fund_id} not found in dataset")
                continue
            
            fund_row = fund_info.iloc[0]
            fund_name = fund_row.get('fund_name', fund_id)
            category = fund_row.get('morningstar_category')
            
            print(f"\nProcessing: {fund_name} (Category: {category})")
            
            # Score peers within the same category
            peer_scores_df = self.score_peers_for_fund(
                fund_id, 
                all_funds_df, 
                exclude_passive=exclude_passive
            )
            
            if peer_scores_df.empty:
                peer_groups[fund_id] = {
                    'fund_name': fund_name,
                    'morningstar_category': category,
                    'peer_count': 0,
                    'peers': []
                }
                continue
            
            # Filter by minimum score and max peers
            qualified_peers = peer_scores_df[peer_scores_df['peer_score'] >= min_score].head(max_peers)
            
            # Store results
            peer_groups[fund_id] = {
                'fund_name': fund_name,
                'morningstar_category': category,
                'peer_count': len(qualified_peers),
                'avg_peer_score': round(qualified_peers['peer_score'].mean(), 2) if not qualified_peers.empty else 0,
                'peers': qualified_peers.to_dict('records'),
                'category_total_funds': len(peer_scores_df) + 1,  # Including the target fund
                'score_distribution': {
                    '90-100': len(peer_scores_df[peer_scores_df['peer_score'] >= 90]),
                    '80-89': len(peer_scores_df[(peer_scores_df['peer_score'] >= 80) & (peer_scores_df['peer_score'] < 90)]),
                    '70-79': len(peer_scores_df[(peer_scores_df['peer_score'] >= 70) & (peer_scores_df['peer_score'] < 80)]),
                    'Below 70': len(peer_scores_df[peer_scores_df['peer_score'] < 70])
                }
            }
            
            # Collect summary statistics
            summary_stats.append({
                'fund_id': fund_id,
                'fund_name': fund_name,
                'category': category,
                'total_in_category': len(peer_scores_df) + 1,
                'qualified_peers': len(qualified_peers),
                'avg_score': round(qualified_peers['peer_score'].mean(), 2) if not qualified_peers.empty else 0
            })
        
        # Print summary if requested
        if output_summary and summary_stats:
            print("\n" + "="*80)
            print("PEER GROUP SUMMARY")
            print("="*80)
            summary_df = pd.DataFrame(summary_stats)
            print(summary_df.to_string(index=False))
            print("\nTotal funds processed:", len(summary_stats))
            print(f"Average qualified peers per fund: {summary_df['qualified_peers'].mean():.1f}")
        
        return peer_groups
    
    def export_peer_groups_to_excel(self, 
                                   peer_groups: Dict,
                                   output_file: str = 'peer_groups.xlsx'):
        """
        Export peer groups to Excel with separate sheets for each fund
        
        Parameters:
        -----------
        peer_groups: Dictionary from create_peer_groups_for_firm_funds
        output_file: Output Excel filename
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for fund_id, data in peer_groups.items():
                summary_data.append({
                    'Fund ID': fund_id,
                    'Fund Name': data['fund_name'],
                    'M* Category': data['morningstar_category'],
                    'Total in Category': data.get('category_total_funds', 0),
                    'Qualified Peers': data['peer_count'],
                    'Avg Peer Score': data.get('avg_peer_score', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual fund sheets
            for fund_id, data in peer_groups.items():
                if data['peers']:
                    peers_df = pd.DataFrame(data['peers'])
                    # Limit sheet name to 31 characters (Excel limit)
                    sheet_name = f"{fund_id[:20]}_peers" if len(fund_id) > 20 else f"{fund_id}_peers"
                    peers_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nPeer groups exported to {output_file}")


# Example usage
def example_usage():
    """
    Demonstrate how to use the FIPeerScoring class with M* categories
    """
    # Sample fund data with Morningstar categories
    funds_data = pd.DataFrame({
        'fund_id': ['FIRM001', 'FIRM002', 'COMP001', 'COMP002', 'COMP003', 
                   'COMP004', 'COMP005', 'COMP006', 'COMP007', 'COMP008'],
        'fund_name': ['Our Bond Fund A', 'Our Bond Fund B', 'Competitor Fund 1', 
                     'Competitor Fund 2', 'Competitor Fund 3', 'Competitor Fund 4',
                     'Competitor Fund 5', 'Competitor Fund 6', 'Competitor Fund 7',
                     'Competitor Fund 8'],
        'morningstar_category': ['US Fund Short-Term Bond', 'US Fund Corporate Bond', 
                                'US Fund Short-Term Bond', 'US Fund Short-Term Bond',
                                'US Fund Corporate Bond', 'US Fund Corporate Bond',
                                'US Fund Short-Term Bond', 'US Fund Corporate Bond',
                                'US Fund Short-Term Bond', 'US Fund Corporate Bond'],
        'currency': ['USD', 'USD', 'USD', 'EUR', 'USD', 'USD', 'USD', 'GBP', 'USD', 'USD'],
        'is_passive': [False, False, False, True, False, True, False, False, True, False],
        'fee_band': [2, 3, 2, 1, 3, 2, 4, 3, 1, 5],
        'region': ['developed', 'developed', 'developed', 'developed', 'emerging', 
                  'developed', 'developed', 'developed', 'developed', 'other'],
        'primary_sector': ['corporate', 'corporate', 'corporate', 'government', 
                          'corporate', 'mixed', 'corporate', 'corporate',
                          'government', 'high yield']
    })
    
    # Your firm's fund IDs
    firm_fund_ids = ['FIRM001', 'FIRM002']
    
    # Initialize scorer
    scorer = FIPeerScoring(
        currency_weight=30,
        passive_weight=10,
        fee_weight=25,
        region_weight=20,
        sector_weight=15
    )
    
    # Create peer groups for your firm's funds
    peer_groups = scorer.create_peer_groups_for_firm_funds(
        firm_fund_ids=firm_fund_ids,
        all_funds_df=funds_data,
        min_score=60,  # Lowered for demo purposes
        max_peers=10,
        exclude_passive=True,  # Exclude passive funds from peer groups
        output_summary=True
    )
    
    # Display detailed results for each fund
    print("\n" + "="*80)
    print("DETAILED PEER GROUPS")
    print("="*80)
    
    for fund_id, peer_data in peer_groups.items():
        print(f"\n{peer_data['fund_name']} ({fund_id})")
        print(f"Category: {peer_data['morningstar_category']}")
        print(f"Qualified Peers: {peer_data['peer_count']}")
        
        if peer_data['peers']:
            print("\nTop Peers:")
            for i, peer in enumerate(peer_data['peers'][:5], 1):
                print(f"  {i}. {peer['fund_name']} - Score: {peer['peer_score']}")
                print(f"     Currency: {peer['currency']}, Fee Band: {peer['fee_band']}, "
                      f"Region: {peer['region']}")
    
    # Export to Excel (optional)
    # scorer.export_peer_groups_to_excel(peer_groups, 'fi_peer_groups.xlsx')


if __name__ == "__main__":
    example_usage()
