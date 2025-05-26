```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class AIProductRoadmapAnalyzer:
    """
    A tool for analyzing and visualizing AI product roadmaps for veterinary applications.
    
    This class provides methods to track progress, analyze resource allocation,
    identify risks, and visualize the product development timeline.
    """
    
    def __init__(self, roadmap_file=None):
        """
        Initialize the roadmap analyzer.
        
        Args:
            roadmap_file: Path to CSV file containing roadmap data (optional)
        """
        self.roadmap_data = None
        self.resources_data = None
        self.risks_data = None
        
        if roadmap_file and os.path.exists(roadmap_file):
            self.load_roadmap(roadmap_file)
    
    def load_roadmap(self, file_path):
        """
        Load roadmap data from CSV file.
        
        Expected CSV format:
        task_id,task_name,phase,start_date,end_date,status,dependencies,owner,priority,complexity
        
        Args:
            file_path: Path to CSV file
        """
        try:
            self.roadmap_data = pd.read_csv(file_path)
            
            # Convert date strings to datetime objects
            self.roadmap_data['start_date'] = pd.to_datetime(self.roadmap_data['start_date'])
            self.roadmap_data['end_date'] = pd.to_datetime(self.roadmap_data['end_date'])
            
            # Calculate duration in days
            self.roadmap_data['duration'] = (self.roadmap_data['end_date'] - 
                                           self.roadmap_data['start_date']).dt.days
            
            print(f"Loaded roadmap with {len(self.roadmap_data)} tasks")
            return True
        except Exception as e:
            print(f"Error loading roadmap: {e}")
            return False
    
    def load_resources(self, file_path):
        """
        Load resource allocation data.
        
        Expected CSV format:
        resource_id,resource_name,role,availability,cost_per_day,skills
        
        Args:
            file_path: Path to CSV file
        """
        try:
            self.resources_data = pd.read_csv(file_path)
            print(f"Loaded resource data with {len(self.resources_data)} resources")
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False
    
    def load_risks(self, file_path):
        """
        Load risk assessment data.
        
        Expected CSV format:
        risk_id,risk_name,category,probability,impact,mitigation,owner
        
        Args:
            file_path: Path to CSV file
        """
        try:
            self.risks_data = pd.read_csv(file_path)
            
            # Calculate risk score
            self.risks_data['risk_score'] = self.risks_data['probability'] * self.risks_data['impact']
            
            print(f"Loaded risk data with {len(self.risks_data)} identified risks")
            return True
        except Exception as e:
            print(f"Error loading risks: {e}")
            return False
    
    def create_sample_data(self, output_dir='./data'):
        """
        Create sample data files for demonstration purposes.
        
        Args:
            output_dir: Directory to save sample files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample roadmap data
        roadmap_data = {
            'task_id': list(range(1, 21)),
            'task_name': [
                'Market Research', 'Stakeholder Interviews', 'Use Case Definition',
                'Data Requirements Analysis', 'Technical Architecture Design',
                'Data Collection Strategy', 'Annotation Protocol Development',
                'Initial Model Selection', 'Data Pipeline Implementation',
                'Baseline Model Training', 'Model Evaluation Framework',
                'Performance Optimization', 'Clinical Integration Design',
                'User Interface Prototyping', 'Alpha Testing Protocol',
                'Regulatory Assessment', 'Beta Deployment Planning',
                'Documentation Development', 'Go-to-Market Strategy',
                'Launch Preparation'
            ],
            'phase': [
                'Discovery', 'Discovery', 'Discovery',
                'Planning', 'Planning',
                'Data Foundation', 'Data Foundation',
                'Model Development', 'Model Development',
                'Model Development', 'Model Development',
                'Model Development', 'Product Integration',
                'Product Integration', 'Validation',
                'Validation', 'Validation',
                'Launch Preparation', 'Launch Preparation',
                'Launch Preparation'
            ],
            'start_date': [
                datetime.now() + timedelta(days=i*14) for i in range(20)
            ],
            'end_date': [
                datetime.now() + timedelta(days=i*14 + 13) for i in range(20)
            ],
            'status': [
                'Completed', 'Completed', 'Completed',
                'Completed', 'In Progress',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started', 'Not Started',
                'Not Started'
            ],
            'dependencies': [
                '', '', '1,2',
                '3', '3,4',
                '4,5', '6',
                '5,7', '7,8',
                '9', '10',
                '10,11', '5,11',
                '13', '13,14',
                '15', '16',
                '12,17', '18',
                '17,18,19'
            ],
            'owner': [
                'Product Manager', 'Product Manager', 'Product Manager',
                'Data Scientist', 'Technical Lead',
                'Data Scientist', 'Data Scientist',
                'ML Engineer', 'ML Engineer',
                'ML Engineer', 'ML Engineer',
                'ML Engineer', 'Software Engineer',
                'UX Designer', 'Product Manager',
                'Regulatory Specialist', 'Technical Lead',
                'Technical Writer', 'Product Manager',
                'Product Manager'
            ],
            'priority': [
                'High', 'High', 'High',
                'High', 'High',
                'High', 'Medium',
                'High', 'High',
                'High', 'Medium',
                'Medium', 'High',
                'Medium', 'High',
                'High', 'Medium',
                'Medium', 'Medium',
                'High'
            ],
            'complexity': [
                2, 2, 3,
                4, 5,
                4, 3,
                4, 5,
                5, 3,
                5, 4,
                3, 4,
                5, 3,
                2, 3,
                4
            ]
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        roadmap_df.to_csv(f"{output_dir}/sample_roadmap.csv", index=False)
        
        # Create sample resources data
        resources_data = {
            'resource_id': list(range(1, 11)),
            'resource_name': [
                'John Smith', 'Maria Garcia', 'Ahmed Hassan',
                'Sarah Johnson', 'Li Wei', 'Carlos Rodriguez',
                'Emma Wilson', 'Raj Patel', 'Olivia Brown',
                'Michael Kim'
            ],
            'role': [
                'Product Manager', 'Data Scientist', 'ML Engineer',
                'Software Engineer', 'UX Designer', 'Technical Lead',
                'Veterinary Specialist', 'ML Engineer', 'Technical Writer',
                'Regulatory Specialist'
            ],
            'availability': [
                0.5, 1.0, 0.8,
                1.0, 0.6, 0.7,
                0.3, 1.0, 0.5,
                0.4
            ],
            'cost_per_day': [
                800, 700, 750,
                700, 650, 850,
                900, 750, 600,
                800
            ],
            'skills': [
                'Product Strategy, Stakeholder Management',
                'Data Analysis, Python, Statistics',
                'TensorFlow, PyTorch, Computer Vision',
                'Python, API Development, System Integration',
                'UI/UX Design, Prototyping',
                'System Architecture, Technical Planning',
                'Veterinary Radiology, Clinical Workflows',
                'NLP, Machine Learning, Python',
                'Technical Documentation, Content Development',
                'Regulatory Affairs, Compliance'
            ]
        }
        
        resources_df = pd.DataFrame(resources_data)
        resources_df.to_csv(f"{output_dir}/sample_resources.csv", index=False)
        
        # Create sample risks data
        risks_data = {
            'risk_id': list(range(1, 11)),
            'risk_name': [
                'Insufficient training data',
                'Poor model generalization across breeds',
                'Clinical integration challenges',
                'Regulatory approval delays',
                'User adoption barriers',
                'Data quality issues',
                'Resource constraints',
                'Technical performance below targets',
                'Competitor product launch',
                'Scope creep'
            ],
            'category': [
                'Data', 'Technical', 'Integration',
                'Regulatory', 'Market', 'Data',
                'Resource', 'Technical', 'Market',
                'Project Management'
            ],
            'probability': [
                0.7, 0.5, 0.6,
                0.4, 0.5, 0.6,
                0.3, 0.4, 0.3,
                0.7
            ],
            'impact': [
                0.8, 0.9, 0.7,
                0.8, 0.6, 0.7,
                0.5, 0.8, 0.6,
                0.5
            ],
            'mitigation': [
                'Establish data sharing partnerships with veterinary teaching hospitals',
                'Implement breed-specific model adaptations and extensive testing',
                'Early engagement with IT teams and workflow analysis',
                'Early consultation with regulatory experts and pre-submission meetings',
                'Involve veterinarians in design process and create comprehensive training',
                'Implement robust data validation and cleaning pipeline',
                'Secure additional budget contingency and flexible resource allocation',
                'Establish performance benchmarks and regular evaluation checkpoints',
                'Accelerate development timeline and focus on unique value propositions',
                'Implement strict change control process and regular scope reviews'
            ],
            'owner': [
                'Data Scientist', 'ML Engineer', 'Software Engineer',
                'Regulatory Specialist', 'Product Manager', 'Data Scientist',
                'Technical Lead', 'ML Engineer', 'Product Manager',
                'Technical Lead'
            ]
        }
        
        risks_df = pd.DataFrame(risks_data)
        risks_df.to_csv(f"{output_dir}/sample_risks.csv", index=False)
        
        print(f"Sample data files created in {output_dir}")
        return {
            'roadmap': f"{output_dir}/sample_roadmap.csv",
            'resources': f"{output_dir}/sample_resources.csv",
            'risks': f"{output_dir}/sample_risks.csv"
        }
    
    def generate_gantt_chart(self, output_file=None, figsize=(12, 8)):
        """
        Generate a Gantt chart visualization of the roadmap.
        
        Args:
            output_file: Path to save the chart image (optional)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure
        """
        if self.roadmap_data is None:
            print("No roadmap data loaded")
            return None
        
        # Sort by start date
        df = self.roadmap_data.sort_values('start_date')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for different phases
        phase_colors = {
            'Discovery': '#1f77b4',
            'Planning': '#ff7f0e',
            'Data Foundation': '#2ca02c',
            'Model Development': '#d62728',
            'Product Integration': '#9467bd',
            'Validation': '#8c564b',
            'Launch Preparation': '#e377c2'
        }
        
        # Define colors for different statuses
        status_colors = {
            'Completed': 1.0,
            'In Progress': 0.5,
            'Not Started': 0.2
        }
        
        # Plot tasks as horizontal bars
        for i, task in df.iterrows():
            phase_color = phase_colors.get(task['phase'], '#7f7f7f')
            alpha = status_colors.get(task['status'], 0.5)
            
            # Plot task bar
            ax.barh(task['task_name'], task['duration'], left=task['start_date'], 
                   color=phase_color, alpha=alpha, height=0.5)
            
            # Add task ID
            ax.text(task['start_date'], task['task_name'], f" {task['task_id']}", 
                   va='center', ha='right', fontsize=8)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add legend for phases
        phase_patches = [plt.Rectangle((0,0), 1, 1, color=color) for color in phase_colors.values()]
        ax.legend(phase_patches, phase_colors.keys(), loc='upper right', title='Phases')
        
        # Add title and labels
        ax.set_title('AI Veterinary Diagnostics Product Roadmap')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tasks')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Gantt chart saved to {output_file}")
        
        return fig
    
    def analyze_critical_path(self):
        """
        Analyze the critical path in the roadmap.
        
        Returns:
            DataFrame containing critical path tasks
        """
        if self.roadmap_data is None:
            print("No roadmap data loaded")
            return None
        
        # Create a dictionary to store task dependencies
        task_dict = {}
        for i, task in self.roadmap_data.iterrows():
            task_dict[str(task['task_id'])] = {
                'name': task['task_name'],
                'duration': task['duration'],
                'dependencies': [dep.strip() for dep in task['dependencies'].split(',') if dep.strip()],
                'start': None,
                'end': None
            }
        
        # Forward pass to calculate earliest start and end times
        for task_id in task_dict:
            self._calculate_earliest_times(task_id, task_dict)
        
        # Find the project end time
        project_end = max(task['end'] for task in task_dict.values() if task['end'] is not None)
        
        # Backward pass to calculate latest start and end times
        for task_id in task_dict:
            task_dict[task_id]['latest_end'] = project_end
            task_dict[task_id]['latest_start'] = project_end
        
        for task_id in reversed(list(task_dict.keys())):
            self._calculate_latest_times(task_id, task_dict)
        
        # Calculate slack and identify critical path
        critical_path = []
        for task_id, task in task_dict.items():
            if task['start'] is not None and task['latest_start'] is not None:
                slack = task['latest_start'] - task['start']
                if slack == 0:
                    critical_path.append({
                        'task_id': task_id,
                        'task_name': task['name'],
                        'duration': task['duration'],
                        'earliest_start': task['start'],
                        'earliest_end': task['end'],
                        'latest_start': task['latest_start'],
                        'latest_end': task['latest_end'],
                        'slack': slack
                    })
        
        # Convert to DataFrame and sort by earliest start
        critical_df = pd.DataFrame(critical_path).sort_values('earliest_start')
        
        return critical_df
    
    def _calculate_earliest_times(self, task_id, task_dict):
        """Helper method for critical path calculation - forward pass"""
        task = task_dict[task_id]
        
        # If start time already calculated, return
        if task['start'] is not None:
            return task['start'], task['end']
        
        # If no dependencies, start at time 0
        if not task['dependencies'] or task['dependencies'] == ['']:
            task['start'] = 0
            task['end'] = task['duration']
            return task['start'], task['end']
        
        # Calculate start time based on dependencies
        max_end_time = 0
        for dep_id in task['dependencies']:
            if dep_id in task_dict:
                _, dep_end = self._calculate_earliest_times(dep_id, task_dict)
                max_end_time = max(max_end_time, dep_end)
        
        task['start'] = max_end_time
        task['end'] = max_end_time + task['duration']
        
        return task['start'], task['end']
    
    def _calculate_latest_times(self, task_id, task_dict):
        """Helper method for critical path calculation - backward pass"""
        task = task_dict[task_id]
        
        # Find all tasks that depend on this task
        dependents = []
        for tid, t in task_dict.items():
            if task_id in t['dependencies']:
                dependents.append(tid)
        
        # If no dependents, latest end time is project end time
        if not dependents:
            return task['latest_start'], task['latest_end']
        
        # Calculate latest end time based on dependents
        min_start_time = float('inf')
        for dep_id in dependents:
            dep_latest_start, _ = self._calculate_latest_times(dep_id, task_dict)
            min_start_time = min(min_start_time, dep_latest_start)
        
        task['latest_end'] = min_start_time
        task['latest_start'] = min_start_time - task['duration']
        
        return task['latest_start'], task['latest_end']
    
    def analyze_resource_allocation(self):
        """
        Analyze resource allocation across the roadmap.
        
        Returns:
            DataFrame with resource allocation analysis
        """
        if self.roadmap_data is None or self.resources_data is None:
            print("Both roadmap and resource data must be loaded")
            return None
        
        # Create a timeline of days
        start_date = self.roadmap_data['start_date'].min()
        end_date = self.roadmap_data['end_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create a dictionary to track resource allocation by day
        resource_allocation = {resource: [0] * len(date_range) for resource in self.resources_data['resource_name']}
        
        # Allocate resources based on task ownership
        for i, task in self.roadmap_data.iterrows():
            owner = task['owner']
            if owner in resource_allocation:
                task_dates = pd.date_range(start=task['start_date'], end=task['end_date'])
                for date in task_dates:
                    if date in date_range:
                        idx = (date - start_date).days
                        if idx < len(resource_allocation[owner]):
                            resource_allocation[owner][idx] += 1
        
        # Convert to DataFrame
        allocation_df = pd.DataFrame(resource_allocation, index=date_range)
        
        # Calculate statistics
        stats = []
        for resource in self.resources_data['resource_name']:
            if resource in allocation_df.columns:
                resource_data = self.resources_data[self.resources_data['resource_name'] == resource].iloc[0]
                availability = resource_data['availability']
                max_allocation = allocation_df[resource].max()
                avg_allocation = allocation_df[resource].mean()
                overallocated_days = (allocation_df[resource] > availability).sum()
                
                stats.append({
                    'resource_name': resource,
                    'role': resource_data['role'],
                    'availability': availability,
                    'max_allocation': max_allocation,
                    'avg_allocation': avg_allocation,
                    'overallocated_days': overallocated_days,
                    'overallocation_percentage': (overallocated_days / len(date_range)) * 100
                })
        
        stats_df = pd.DataFrame(stats)
        return stats_df
    
    def visualize_resource_allocation(self, output_file=None, figsize=(14, 8)):
        """
        Visualize resource allocation over time.
        
        Args:
            output_file: Path to save the chart image (optional)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure
        """
        if self.roadmap_data is None or self.resources_data is None:
            print("Both roadmap and resource data must be loaded")
            return None
        
        # Create a timeline of days
        start_date = self.roadmap_data['start_date'].min()
        end_date = self.roadmap_data['end_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create a dictionary to track resource allocation by day
        resource_allocation = {resource: [0] * len(date_range) for resource in self.resources_data['resource_name']}
        
        # Allocate resources based on task ownership
        for i, task in self.roadmap_data.iterrows():
            owner = task['owner']
            if owner in resource_allocation:
                task_dates = pd.date_range(start=task['start_date'], end=task['end_date'])
                for date in task_dates:
                    if date in date_range:
                        idx = (date - start_date).days
                        if idx < len(resource_allocation[owner]):
                            resource_allocation[owner][idx] += 1
        
        # Convert to DataFrame
        allocation_df = pd.DataFrame(resource_allocation, index=date_range)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot resource allocation
        for resource in allocation_df.columns:
            ax.plot(allocation_df.index, allocation_df[resource], label=resource)
        
        # Add availability lines
        for resource in self.resources_data['resource_name']:
            if resource in allocation_df.columns:
                availability = self.resources_data[self.resources_data['resource_name'] == resource]['availability'].iloc[0]
                ax.axhline(y=availability, linestyle='--', color='gray', alpha=0.5)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add legend, title and labels
        ax.legend(loc='upper right')
        ax.set_title('Resource Allocation Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation Level')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Resource allocation chart saved to {output_file}")
        
        return fig
    
    def analyze_risks(self):
        """
        Analyze and prioritize risks.
        
        Returns:
            DataFrame with risk analysis
        """
        if self.risks_data is None:
            print("No risk data loaded")
            return None
        
        # Sort risks by risk score
        sorted_risks = self.risks_data.sort_values('risk_score', ascending=False)
        
        # Categorize risks
        sorted_risks['risk_level'] = pd.cut(
            sorted_risks['risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return sorted_risks
    
    def visualize_risk_matrix(self, output_file=None, figsize=(10, 8)):
        """
        Create a risk matrix visualization.
        
        Args:
            output_file: Path to save the chart image (optional)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure
        """
        if self.risks_data is None:
            print("No risk data loaded")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        scatter = ax.scatter(
            self.risks_data['probability'],
            self.risks_data['impact'],
            s=self.risks_data['risk_score'] * 500,  # Size based on risk score
            c=self.risks_data['risk_score'],  # Color based on risk score
            cmap='YlOrRd',
            alpha=0.7
        )
        
        # Add risk labels
        for i, risk in self.risks_data.iterrows():
            ax.annotate(
                f"R{risk['risk_id']}",
                (risk['probability'], risk['impact']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Risk Score')
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add risk zones
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)
        
        # Add zone labels
        ax.text(0.25, 0.25, 'Low Risk', ha='center', va='center', alpha=0.5)
        ax.text(0.75, 0.25, 'Medium Risk', ha='center', va='center', alpha=0.5)
        ax.text(0.25, 0.75, 'Medium Risk', ha='center', va='center', alpha=0.5)
        ax.text(0.75, 0.75, 'High Risk', ha='center', va='center', alpha=0.5)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add title and labels
        ax.set_title('Risk Assessment Matrix')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Impact')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Risk matrix saved to {output_file}")
        
        return fig
    
    def generate_roadmap_report(self, output_file=None):
        """
        Generate a comprehensive roadmap analysis report.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            Report text
        """
        report = []
        
        # Add header
        report.append("# AI Veterinary Diagnostics Product Roadmap Analysis")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        # Roadmap summary
        if self.roadmap_data is not None:
            report.append("## Roadmap Summary")
            report.append(f"- Total tasks: {len(self.roadmap_data)}")
            report.append(f"- Start date: {self.roadmap_data['start_date'].min().strftime('%Y-%m-%d')}")
            report.append(f"- End date: {self.roadmap_data['end_date'].max().strftime('%Y-%m-%d')}")
            report.append(f"- Total duration: {(self.roadmap_data['end_date'].max() - self.roadmap_data['start_date'].min()).days} days")
            
            # Task status
            status_counts = self.roadmap_data['status'].value_counts()
            report.append("\n### Task Status")
            for status, count in status_counts.items():
                report.append(f"- {status}: {count} tasks ({count/len(self.roadmap_data)*100:.1f}%)")
            
            # Phase breakdown
            phase_counts = self.roadmap_data['phase'].value_counts()
            report.append("\n### Phase Breakdown")
            for phase, count in phase_counts.items():
                report.append(f"- {phase}: {count} tasks")
        
        # Critical path analysis
        report.append("\n## Critical Path Analysis")
        try:
            critical_path = self.analyze_critical_path()
            if critical_path is not None and not critical_path.empty:
                report.append("\nThe following tasks are on the critical path and require careful monitoring:")
                for i, task in critical_path.iterrows():
                    report.append(f"- Task {task['task_id']}: {task['task_name']} (Duration: {task['duration']} days)")
                
                report.append("\nAny delay in these tasks will directly impact the project end date.")
            else:
                report.append("\nNo critical path analysis available.")
        except Exception as e:
            report.append(f"\nError in critical path analysis: {e}")
        
        # Resource allocation
        if self.resources_data is not None:
            report.append("\n## Resource Allocation Analysis")
            try:
                resource_analysis = self.analyze_resource_allocation()
                if resource_analysis is not None and not resource_analysis.empty:
                    # Identify overallocated resources
                    overallocated = resource_analysis[resource_analysis['overallocation_percentage'] > 0]
                    if not overallocated.empty:
                        report.append("\n### Overallocated Resources")
                        for i, resource in overallocated.iterrows():
                            report.append(f"- {resource['resource_name']} ({resource['role']}): "
                                         f"Overallocated for {resource['overallocated_days']} days "
                                         f"({resource['overallocation_percentage']:.1f}% of project duration)")
                    
                    # Resource utilization
                    report.append("\n### Resource Utilization")
                    for i, resource in resource_analysis.iterrows():
                        utilization = (resource['avg_allocation'] / resource['availability']) * 100
                        report.append(f"- {resource['resource_name']} ({resource['role']}): "
                                     f"{utilization:.1f}% average utilization")
                else:
                    report.append("\nNo resource allocation analysis available.")
            except Exception as e:
                report.append(f"\nError in resource allocation analysis: {e}")
        
        # Risk assessment
        if self.risks_data is not None:
            report.append("\n## Risk Assessment")
            try:
                risk_analysis = self.analyze_risks()
                if risk_analysis is not None and not risk_analysis.empty:
                    # High risks
                    high_risks = risk_analysis[risk_analysis['risk_level'] == 'High']
                    if not high_risks.empty:
                        report.append("\n### High Priority Risks")
                        for i, risk in high_risks.iterrows():
                            report.append(f"- R{risk['risk_id']}: {risk['risk_name']} "
                                         f"(Score: {risk['risk_score']:.2f})")
                            report.append(f"  - Mitigation: {risk['mitigation']}")
                    
                    # Risk distribution
                    risk_levels = risk_analysis['risk_level'].value_counts()
                    report.append("\n### Risk Distribution")
                    for level, count in risk_levels.items():
                        report.append(f"- {level} risks: {count}")
                else:
                    report.append("\nNo risk analysis available.")
            except Exception as e:
                report.append(f"\nError in risk analysis: {e}")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        # Add recommendations based on analyses
        recommendations = []
        
        # Critical path recommendations
        try:
            if self.roadmap_data is not None:
                critical_path = self.analyze_critical_path()
                if critical_path is not None and not critical_path.empty:
                    recommendations.append("Focus management attention on critical path tasks to prevent project delays.")
        except:
            pass
        
        # Resource recommendations
        try:
            if self.resources_data is not None:
                resource_analysis = self.analyze_resource_allocation()
                if resource_analysis is not None and not resource_analysis.empty:
                    overallocated = resource_analysis[resource_analysis['overallocation_percentage'] > 20]
                    if not overallocated.empty:
                        recommendations.append(f"Address resource overallocation for {len(overallocated)} team members by redistributing tasks or adding resources.")
        except:
            pass
        
        # Risk recommendations
        try:
            if self.risks_data is not None:
                risk_analysis = self.analyze_risks()
                if risk_analysis is not None and not risk_analysis.empty:
                    high_risks = risk_analysis[risk_analysis['risk_level'] == 'High']
                    if not high_risks.empty:
                        recommendations.append(f"Develop detailed mitigation plans for the {len(high_risks)} high-priority risks identified.")
        except:
            pass
        
        # Add general recommendations
        recommendations.extend([
            "Review and update the roadmap regularly as the project progresses.",
            "Conduct bi-weekly status reviews focusing on critical path tasks.",
            "Establish clear communication channels for risk reporting and mitigation.",
            "Consider adding buffer time to high-risk or complex tasks.",
            "Ensure veterinary domain experts are engaged throughout the development process."
        ])
        
        # Add recommendations to report
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        # Conclusion
        report.append("\n## Conclusion")
        report.append("This roadmap analysis provides a comprehensive overview of the AI Veterinary Diagnostics product development plan. By focusing on critical path management, addressing resource constraints, and proactively mitigating identified risks, the team can increase the likelihood of successful and timely product delivery.")
        
        # Join report sections
        report_text = "\n".join(report)
        
        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = AIProductRoadmapAnalyzer()
    
    # Create sample data
    sample_files = analyzer.create_sample_data('./data')
    
    # Load sample data
    analyzer.load_roadmap(sample_files['roadmap'])
    analyzer.load_resources(sample_files['resources'])
    analyzer.load_risks(sample_files['risks'])
    
    # Generate visualizations
    analyzer.generate_gantt_chart('./output/gantt_chart.png')
    analyzer.visualize_resource_allocation('./output/resource_allocation.png')
    analyzer.visualize_risk_matrix('./output/risk_matrix.png')
    
    # Generate report
    analyzer.generate_roadmap_report('./output/roadmap_analysis_report.md')
    
    print("Analysis completed successfully")
```
