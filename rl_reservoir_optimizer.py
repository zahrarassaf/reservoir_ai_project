import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== FINAL SPE9 ENVIRONMENT ====================
class FinalSPE9Env:
    
    def __init__(self, uncertainty_level=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_wells = 4
        self.well_names = ['P1', 'P2', 'P3', 'P4']
        
        self.base_productivity = np.array([100, 90, 110, 95])
        self.productivity_uncertainty = uncertainty_level
        
        self.base_pore_volume = 50e6
        self.base_initial_oil = 30e6
        self.pore_volume_uncertainty = uncertainty_level
        self.oil_in_place_uncertainty = uncertainty_level
        
        self._randomize_parameters()
        
        self.initial_pressure = 3600.0
        self.min_bhp = 1500.0
        self.compressibility = 1e-5
        self.formation_volume_factor = 1.2
        self.oil_viscosity = 2.0
        
        self.max_oil_rate_per_well = 2000.0
        self.max_total_oil_rate = 8000.0
        
        self.oil_price = 70.0
        self.op_cost = 25.0
        self.discount_rate = 0.10
        
        self.days_per_step = 30
        self.max_simulation_years = 15
        
        self.water_cut_growth_rate = 0.0008
        self.max_water_cut = 0.90
        
        self.residual_oil_saturation = 0.20
        self.aquifer_strength = np.random.uniform(0.1, 0.3)
        
        self.state_dim = 3 + self.n_wells + 2
        self.action_dim = self.n_wells
        
        self.reset()
    
    def _randomize_parameters(self):
        self.well_productivity = self.base_productivity * np.random.uniform(
            1 - self.productivity_uncertainty, 
            1 + self.productivity_uncertainty, 
            size=self.base_productivity.shape
        )
        
        self.pore_volume = self.base_pore_volume * np.random.uniform(
            1 - self.pore_volume_uncertainty, 
            1 + self.pore_volume_uncertainty
        )
        
        self.initial_oil = self.base_initial_oil * np.random.uniform(
            1 - self.oil_in_place_uncertainty, 
            1 + self.oil_in_place_uncertainty
        )
    
    def reset(self, new_realization=False):
        if new_realization:
            self._randomize_parameters()
            self.aquifer_strength = np.random.uniform(0.1, 0.3)
        
        self.pressure = self.initial_pressure
        self.oil_in_place = self.initial_oil
        self.days = 0
        self.oil_rates = np.zeros(self.n_wells)
        self.water_cut = 0.05
        self.cumulative_oil = 0.0
        self.cumulative_water = 0.0
        self.recovery_factor = 0.0
        
        self.history = {
            'pressure': [], 'oil_rates': [], 'total_oil_rate': [],
            'cumulative_oil': [], 'recovery': [], 'actions': [],
            'water_cut': [], 'days': [], 'water_rates': []
        }
        
        return self._get_state()
    
    def _get_state(self):
        state = [
            self.pressure / self.initial_pressure,
            self.oil_in_place / self.initial_oil,
            min(self.days / (self.max_simulation_years * 365.0), 1.0),
            self.water_cut,
            self.recovery_factor / 100.0
        ]
        
        for rate in self.oil_rates:
            state.append(rate / self.max_oil_rate_per_well)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_well_rate(self, well_idx, bhp):
        bhp = max(bhp, self.min_bhp)
        bhp = min(bhp, self.pressure * 0.95)
        
        drawdown = max(self.pressure - bhp, 0.0)
        
        if drawdown <= 0:
            return 0.0
        
        base_pi = self.well_productivity[well_idx]
        pressure_ratio = self.pressure / self.initial_pressure
        pressure_factor = 0.3 + 0.7 * pressure_ratio
        oil_factor = 0.3 + 0.7 * (self.oil_in_place / self.initial_oil)
        
        max_drawdown = self.pressure - self.min_bhp
        if max_drawdown > 0:
            drawdown_ratio = drawdown / max_drawdown
            if drawdown_ratio > 0.6:
                vogel_factor = 1.0 - 0.2 * drawdown_ratio - 0.8 * drawdown_ratio**2
                pressure_factor *= max(vogel_factor, 0.2)
        
        effective_pi = base_pi * pressure_factor * oil_factor
        rate = effective_pi * drawdown
        rate = min(rate, self.max_oil_rate_per_well)
        
        return max(rate, 0.0)
    
    def step(self, actions):
        bhp_targets = 1500.0 + 2100.0 * np.clip(actions, 0.0, 1.0)
        
        well_rates = np.zeros(self.n_wells)
        for i in range(self.n_wells):
            well_rates[i] = self._calculate_well_rate(i, bhp_targets[i])
        
        total_oil_rate = np.sum(well_rates)
        
        if total_oil_rate > self.max_total_oil_rate:
            scaling = self.max_total_oil_rate / total_oil_rate
            well_rates *= scaling
            total_oil_rate = self.max_total_oil_rate
        
        oil_produced = total_oil_rate * self.days_per_step
        
        max_recoverable = self.initial_oil * (1 - self.residual_oil_saturation)
        remaining_recoverable = max(0.0, max_recoverable - self.cumulative_oil)
        
        if oil_produced > remaining_recoverable:
            scaling = remaining_recoverable / oil_produced if oil_produced > 0 else 0.0
            oil_produced *= scaling
            total_oil_rate *= scaling
            well_rates *= scaling
        
        water_rate = total_oil_rate * self.water_cut / (1 - self.water_cut + 1e-10)
        water_produced = water_rate * self.days_per_step
        
        self.oil_in_place -= oil_produced
        self.cumulative_oil += oil_produced
        self.cumulative_water += water_produced
        
        total_voidage = (oil_produced * self.formation_volume_factor) + water_produced
        
        if self.pore_volume > 0:
            dp_production = -total_voidage / (self.pore_volume * self.compressibility)
            dp_aquifer = -dp_production * self.aquifer_strength
            dp_total = dp_production + dp_aquifer
            
            self.pressure += dp_total
            self.pressure = max(self.pressure, self.min_bhp * 0.7)
        
        recovery_ratio = self.cumulative_oil / self.initial_oil
        water_cut_increase = self.water_cut_growth_rate * (1 + 5 * recovery_ratio**2)
        self.water_cut += water_cut_increase
        self.water_cut = min(self.water_cut, self.max_water_cut)
        
        self.days += self.days_per_step
        
        self.oil_rates = well_rates.copy()
        
        self.recovery_factor = (self.cumulative_oil / self.initial_oil) * 100.0
        
        self.history['pressure'].append(self.pressure)
        self.history['oil_rates'].append(well_rates.copy())
        self.history['total_oil_rate'].append(total_oil_rate)
        self.history['cumulative_oil'].append(self.cumulative_oil)
        self.history['recovery'].append(self.recovery_factor)
        self.history['actions'].append(bhp_targets.copy())
        self.history['water_cut'].append(self.water_cut)
        self.history['water_rates'].append(water_rate)
        self.history['days'].append(self.days)
        
        reward = self._calculate_reward(total_oil_rate, oil_produced, water_produced)
        
        done = (self.days >= self.max_simulation_years * 365 or
                self.recovery_factor >= 45.0 or
                total_oil_rate < 50.0 or
                self.oil_in_place <= self.initial_oil * self.residual_oil_saturation)
        
        info = {
            'days': self.days, 'years': self.days / 365.0,
            'oil_rate': total_oil_rate, 'well_rates': well_rates.copy(),
            'water_rate': water_rate, 'cumulative_oil': self.cumulative_oil,
            'cumulative_water': self.cumulative_water, 'recovery': self.recovery_factor,
            'pressure': self.pressure, 'oil_in_place': self.oil_in_place,
            'water_cut': self.water_cut, 'bhp_targets': bhp_targets.copy(),
            'aquifer_strength': self.aquifer_strength
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, oil_rate, oil_produced, water_produced):
        revenue = oil_produced * self.oil_price
        costs = oil_produced * self.op_cost + water_produced * 3.0
        profit = (revenue - costs) / 1000.0
        
        years = self.days / 365.0
        discount = 1.0 / ((1.0 + self.discount_rate) ** years)
        
        reward = profit * discount
        
        if oil_rate > 500:
            reward += np.log1p(oil_rate / 500.0) * 2.0
        
        reward += self.recovery_factor * 3.0
        
        pressure_ratio = self.pressure / self.initial_pressure
        if pressure_ratio > 0.4:
            reward += pressure_ratio * 2.0
        
        if self.pressure < self.min_bhp * 1.1:
            reward -= 1.0
        
        if self.water_cut > 0.7:
            reward -= (self.water_cut - 0.7) * 5.0
        
        return reward

# ==================== DEEP Q-NETWORK AGENT ====================
class DQNAgent:
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 5)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 5)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = deque(maxlen=10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_every = 100
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.action_values = np.linspace(0.2, 0.8, 5)
        
        self.total_steps = 0
        self.training_losses = []
    
    def select_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            action_indices = np.random.randint(0, len(self.action_values), size=self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_values = q_values.view(1, self.action_dim, len(self.action_values))
                action_indices = torch.argmax(q_values, dim=2).numpy()[0]
        
        actions = self.action_values[action_indices]
        
        return actions
    
    def train(self, env, episodes=100, steps_per_episode=120):
        print(f"\nTraining DQN agent for {episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                action = self.select_action(state, explore=True)
                
                next_state, reward, done, info = env.step(action)
                
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self._update_network()
                    self.training_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}/{episodes}: "
                      f"Reward = {episode_reward:.0f}, "
                      f"Avg Reward (last 10) = {avg_reward:.0f}, "
                      f"Epsilon = {self.epsilon:.3f}")
        
        print("Training completed!")
        return episode_rewards
    
    def _update_network(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.q_network(states)
        q_values = q_values.view(self.batch_size, self.action_dim, len(self.action_values))
        
        action_indices = torch.zeros((self.batch_size, self.action_dim), dtype=torch.long)
        for i in range(self.batch_size):
            for j in range(self.action_dim):
                idx = np.argmin(np.abs(self.action_values - actions[i][j]))
                action_indices[i, j] = idx
        
        current_q = q_values.gather(2, action_indices.unsqueeze(2)).squeeze(2)
        current_q = current_q.mean(dim=1, keepdim=True)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.view(self.batch_size, self.action_dim, len(self.action_values))
            next_q_max = next_q_values.max(dim=2)[0].mean(dim=1, keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * next_q_max
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.total_steps += 1
        if self.total_steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

# ==================== ADAPTIVE CONTROLLER ====================
class AdaptiveController:
    
    def get_action(self, pressure_ratio, recovery_factor):
        if recovery_factor < 15:
            if pressure_ratio > 0.7:
                return np.array([0.3, 0.3, 0.3, 0.3])
            elif pressure_ratio > 0.5:
                return np.array([0.4, 0.4, 0.4, 0.4])
            else:
                return np.array([0.5, 0.5, 0.5, 0.5])
        elif recovery_factor < 30:
            if pressure_ratio > 0.6:
                return np.array([0.4, 0.4, 0.4, 0.4])
            elif pressure_ratio > 0.4:
                return np.array([0.5, 0.5, 0.5, 0.5])
            else:
                return np.array([0.6, 0.6, 0.6, 0.6])
        else:
            if pressure_ratio > 0.5:
                return np.array([0.5, 0.5, 0.5, 0.5])
            elif pressure_ratio > 0.3:
                return np.array([0.7, 0.7, 0.7, 0.7])
            else:
                return np.array([0.9, 0.9, 0.9, 0.9])

# ==================== OPTIMIZATION SYSTEM ====================
class SPE9OptimizationSystem:
    
    def __init__(self):
        self.results_dir = Path("results/final_spe9")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_simulation(self, strategy='adaptive', uncertainty=0.1, seed=None):
        print("=" * 60)
        print(f"SPE9 SIMULATION - {strategy.upper()} STRATEGY")
        print("=" * 60)
        
        env = FinalSPE9Env(uncertainty_level=uncertainty, seed=seed)
        
        if strategy == 'adaptive':
            controller = AdaptiveController()
        elif strategy == 'aggressive':
            controller = None
        elif strategy == 'conservative':
            controller = None
        elif strategy == 'rl':
            agent = DQNAgent(env.state_dim, env.action_dim)
            controller = AdaptiveController()
        else:
            controller = AdaptiveController()
        
        env.reset()
        results = []
        
        for year in range(10):
            year_oil = 0
            year_reward = 0
            
            for month in range(12):
                if strategy == 'adaptive':
                    pressure_ratio = env.pressure / env.initial_pressure
                    action = controller.get_action(pressure_ratio, env.recovery_factor)
                elif strategy == 'aggressive':
                    action = np.array([0.2, 0.2, 0.2, 0.2])
                elif strategy == 'conservative':
                    action = np.array([0.8, 0.8, 0.8, 0.8])
                elif strategy == 'rl':
                    state = env._get_state()
                    action = agent.select_action(state, explore=False)
                else:
                    action = np.array([0.5, 0.5, 0.5, 0.5])
                
                state, reward, done, info = env.step(action)
                year_oil += info['oil_rate'] * 30 / 1000
                year_reward += reward
                
                if done:
                    break
            
            results.append({
                'year': year + 1,
                'oil_mbbl': year_oil,
                'cumulative_oil_mmbbl': info['cumulative_oil'] / 1e6,
                'recovery': info['recovery'],
                'pressure': info['pressure'],
                'oil_rate': info['oil_rate'],
                'water_cut': info['water_cut'] * 100
            })
            
            print(f"Year {year+1}: {year_oil:.0f} Mbbl, "
                  f"Recovery={info['recovery']:.1f}%, "
                  f"Pressure={info['pressure']:.0f} psi")
            
            if done:
                break
        
        final_info = {
            'strategy': strategy,
            'total_oil_mmbbl': env.cumulative_oil / 1e6,
            'final_recovery': env.recovery_factor,
            'final_oil_rate': info['oil_rate'],
            'final_pressure': info['pressure'],
            'final_water_cut': info['water_cut'] * 100,
            'aquifer_strength': info['aquifer_strength'],
            'initial_oil_mmbbl': env.initial_oil / 1e6,
            'well_productivity': env.well_productivity.tolist()
        }
        
        print(f"\nFinal Results:")
        print(f"Total Oil Produced: {final_info['total_oil_mmbbl']:.2f} MMbbl")
        print(f"Final Recovery: {final_info['final_recovery']:.1f}%")
        print(f"Final Oil Rate: {final_info['final_oil_rate']:.0f} bpd")
        print(f"Final Pressure: {final_info['final_pressure']:.0f} psi")
        print(f"Final Water Cut: {final_info['final_water_cut']:.1f}%")
        
        self._save_results(final_info, results, strategy)
        
        return final_info, results
    
    def run_comparison_study(self, n_realizations=10):
        print("=" * 60)
        print("STRATEGY COMPARISON STUDY")
        print("=" * 60)
        
        strategies = ['adaptive', 'aggressive', 'conservative']
        all_results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy} strategy...")
            
            strategy_results = []
            for i in range(n_realizations):
                print(f"  Realization {i+1}/{n_realizations}...", end='\r')
                final_info, _ = self.run_single_simulation(
                    strategy=strategy, 
                    uncertainty=0.15,
                    seed=i
                )
                strategy_results.append(final_info)
            
            all_results[strategy] = strategy_results
        
        self._analyze_comparison(all_results)
        
        self._save_comparison_results(all_results)
        
        return all_results
    
    def _analyze_comparison(self, all_results):
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        print(f"\n{'Strategy':<15} {'Recovery (%)':<15} {'Oil (MMbbl)':<15} {'Final Rate (bpd)':<15}")
        print(f"{'':<15} {'Mean ± Std':<15} {'Mean ± Std':<15} {'Mean ± Std':<15}")
        print("-" * 60)
        
        for strategy, results in all_results.items():
            recoveries = [r['final_recovery'] for r in results]
            oils = [r['total_oil_mmbbl'] for r in results]
            rates = [r['final_oil_rate'] for r in results]
            
            recovery_mean = np.mean(recoveries)
            recovery_std = np.std(recoveries)
            oil_mean = np.mean(oils)
            oil_std = np.std(oils)
            rate_mean = np.mean(rates)
            rate_std = np.std(rates)
            
            print(f"{strategy:<15} {recovery_mean:.1f} ± {recovery_std:.1f}{'':<5}"
                  f"{oil_mean:.2f} ± {oil_std:.2f}{'':<5}"
                  f"{rate_mean:.0f} ± {rate_std:.0f}")
        
        best_strategy = None
        best_recovery = 0
        
        for strategy, results in all_results.items():
            avg_recovery = np.mean([r['final_recovery'] for r in results])
            if avg_recovery > best_recovery:
                best_recovery = avg_recovery
                best_strategy = strategy
        
        print(f"\nBest Strategy: {best_strategy} with average recovery of {best_recovery:.1f}%")
        
        self._plot_comparison(all_results)
    
    def _plot_comparison(self, all_results):
        strategies = list(all_results.keys())
        
        recovery_data = []
        oil_data = []
        rate_data = []
        
        for strategy in strategies:
            strategy_results = all_results[strategy]
            
            recoveries = [r['final_recovery'] for r in strategy_results]
            oils = [r['total_oil_mmbbl'] for r in strategy_results]
            rates = [r['final_oil_rate'] for r in strategy_results]
            
            recovery_data.append(recoveries)
            oil_data.append(oils)
            rate_data.append(rates)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        bp1 = axes[0].boxplot(recovery_data, labels=strategies, patch_artist=True)
        axes[0].set_ylabel('Recovery Factor (%)')
        axes[0].set_title('Recovery Factor Comparison')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        bp2 = axes[1].boxplot(oil_data, labels=strategies, patch_artist=True)
        axes[1].set_ylabel('Total Oil (MMbbl)')
        axes[1].set_title('Total Oil Production Comparison')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        bp3 = axes[2].boxplot(rate_data, labels=strategies, patch_artist=True)
        axes[2].set_ylabel('Final Oil Rate (bpd)')
        axes[2].set_title('Final Oil Rate Comparison')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for bplot in [bp1, bp2, bp3]:
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        
        plt.suptitle('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.results_dir / "strategy_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plot saved to {save_path}")
    
    def _save_comparison_results(self, all_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_data = {
            'timestamp': timestamp,
            'n_realizations': len(next(iter(all_results.values()))),
            'strategies': {}
        }
        
        for strategy, results in all_results.items():
            comparison_data['strategies'][strategy] = {
                'average_recovery': np.mean([r['final_recovery'] for r in results]),
                'average_oil': np.mean([r['total_oil_mmbbl'] for r in results]),
                'average_rate': np.mean([r['final_oil_rate'] for r in results]),
                'std_recovery': np.std([r['final_recovery'] for r in results]),
                'std_oil': np.std([r['total_oil_mmbbl'] for r in results]),
                'std_rate': np.std([r['final_oil_rate'] for r in results]),
                'all_results': results
            }
        
        filename = f"comparison_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Comparison results saved to {filepath}")
        return filepath
    
    def _save_results(self, final_info, results, strategy):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            'timestamp': timestamp,
            'strategy': strategy,
            'final_results': final_info,
            'yearly_results': results
        }
        
        filename = f"results_{strategy}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
        
        return filepath

# ==================== VISUALIZATION ====================
def plot_production_history(env, title="SPE9 Production History", save_path=None):
    if not env.history['total_oil_rate']:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    time_years = [d/365.0 for d in env.history['days']]
    
    axes[0, 0].plot(time_years, env.history['total_oil_rate'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Oil Rate (bpd)')
    axes[0, 0].set_title('Oil Production Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_years, np.array(env.history['cumulative_oil'])/1e6, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('Cumulative Oil (MMbbl)')
    axes[0, 1].set_title(f'Cumulative Production: {env.cumulative_oil/1e6:.2f} MMbbl')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(time_years, env.history['recovery'], 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Time (years)')
    axes[0, 2].set_ylabel('Recovery (%)')
    axes[0, 2].set_title(f'Recovery Factor: {env.recovery_factor:.1f}%')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time_years, env.history['pressure'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=env.min_bhp, color='orange', linestyle='--', label='Min BHP')
    axes[1, 0].set_xlabel('Time (years)')
    axes[1, 0].set_ylabel('Pressure (psi)')
    axes[1, 0].set_title('Reservoir Pressure')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_years, env.history['water_cut'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Time (years)')
    axes[1, 1].set_ylabel('Water Cut')
    axes[1, 1].set_title('Field Water Cut')
    axes[1, 1].grid(True, alpha=0.3)
    
    if env.history['actions']:
        actions_array = np.array(env.history['actions'])
        for i in range(min(4, actions_array.shape[1])):
            axes[1, 2].plot(time_years, actions_array[:, i], 
                           label=f'Well {i+1}', alpha=0.8, linewidth=1.5)
        axes[1, 2].set_xlabel('Time (years)')
        axes[1, 2].set_ylabel('BHP (psi)')
        axes[1, 2].set_title('Well Control Strategy')
        axes[1, 2].legend(loc='upper right')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ==================== MAIN INTERFACE ====================
def main():
    print("=" * 80)
    print("SPE9 RESERVOIR OPTIMIZATION SYSTEM - FINAL VERSION")
    print("=" * 80)
    
    system = SPE9OptimizationSystem()
    
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("1. Run single simulation (Adaptive strategy)")
        print("2. Run single simulation (Aggressive strategy)")
        print("3. Run single simulation (Conservative strategy)")
        print("4. Compare all strategies")
        print("5. Run RL training")
        print("6. View results summary")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            final_info, results = system.run_single_simulation(strategy='adaptive')
            
            env = FinalSPE9Env()
            env.reset()
            controller = AdaptiveController()
            for year in range(10):
                for month in range(12):
                    pressure_ratio = env.pressure / env.initial_pressure
                    action = controller.get_action(pressure_ratio, env.recovery_factor)
                    state, reward, done, info = env.step(action)
                    if done:
                        break
                if done:
                    break
            
            fig = plot_production_history(env, title="SPE9 Production - Adaptive Strategy")
            if fig:
                plt.savefig(system.results_dir / "adaptive_strategy.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Plot saved to {system.results_dir}/adaptive_strategy.png")
        
        elif choice == '2':
            final_info, results = system.run_single_simulation(strategy='aggressive')
        
        elif choice == '3':
            final_info, results = system.run_single_simulation(strategy='conservative')
        
        elif choice == '4':
            all_results = system.run_comparison_study(n_realizations=5)
        
        elif choice == '5':
            print("\n" + "="*60)
            print("RL TRAINING DEMO")
            print("="*60)
            
            env = FinalSPE9Env(uncertainty_level=0.1)
            agent = DQNAgent(env.state_dim, env.action_dim)
            
            episodes = 20
            rewards = agent.train(env, episodes=episodes, steps_per_episode=120)
            
            print(f"\nRL training completed for {episodes} episodes!")
            print(f"Final epsilon: {agent.epsilon:.3f}")
            print(f"Average reward: {np.mean(rewards):.0f}")
            
            print("\nTesting trained agent...")
            env.reset()
            total_reward = 0
            
            for step in range(120):
                state = env._get_state()
                action = agent.select_action(state, explore=False)
                state, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            print(f"Test Results:")
            print(f"Total Reward: {total_reward:.0f}")
            print(f"Final Recovery: {info['recovery']:.1f}%")
            print(f"Total Oil: {info['cumulative_oil']/1e6:.2f} MMbbl")
        
        elif choice == '6':
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            
            result_files = list(system.results_dir.glob("results_*.json"))
            
            if not result_files:
                print("No results found. Run some simulations first.")
            else:
                print(f"Found {len(result_files)} result files:")
                
                for file in result_files[-5:]:
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            strategy = data.get('strategy', 'unknown')
                            recovery = data['final_results']['final_recovery']
                            oil = data['final_results']['total_oil_mmbbl']
                            print(f"  {file.stem}: {strategy}, Recovery={recovery:.1f}%, Oil={oil:.2f} MMbbl")
                    except:
                        pass
        
        elif choice == '7':
            print("\nThank you for using SPE9 Reservoir Optimization System!")
            print(f"Results are saved in: {system.results_dir}")
            break
        
        else:
            print("Invalid choice. Please try again.")

# ==================== QUICK DEMO ====================
def quick_demo():
    print("=" * 80)
    print("SPE9 QUICK DEMONSTRATION")
    print("=" * 80)
    
    env = FinalSPE9Env(uncertainty_level=0.1, seed=42)
    
    controller = AdaptiveController()
    
    print("\nRunning adaptive control strategy for 10 years...")
    
    env.reset()
    yearly_results = []
    
    for year in range(10):
        year_oil = 0
        
        for month in range(12):
            pressure_ratio = env.pressure / env.initial_pressure
            action = controller.get_action(pressure_ratio, env.recovery_factor)
            
            state, reward, done, info = env.step(action)
            year_oil += info['oil_rate'] * 30 / 1000
            
            if done:
                break
        
        yearly_results.append({
            'year': year + 1,
            'oil_mbbl': year_oil,
            'cumulative_recovery': info['recovery'],
            'pressure': info['pressure'],
            'water_cut': info['water_cut'] * 100
        })
        
        print(f"Year {year+1}: {year_oil:.0f} Mbbl, "
              f"Recovery={info['recovery']:.1f}%, "
              f"Pressure={info['pressure']:.0f} psi")
        
        if done:
            break
    
    print(f"\nFinal Results:")
    print(f"Total Oil Produced: {env.cumulative_oil/1e6:.2f} MMbbl")
    print(f"Final Recovery: {env.recovery_factor:.1f}%")
    print(f"Final Oil Rate: {info['oil_rate']:.0f} bpd")
    print(f"Final Pressure: {info['pressure']:.0f} psi")
    print(f"Final Water Cut: {info['water_cut']*100:.1f}%")
    
    fig = plot_production_history(env, title="SPE9 Quick Demo - Adaptive Control")
    if fig:
        plt.savefig("spe9_quick_demo_results.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nPlot saved to: spe9_quick_demo_results.png")
    
    return env, yearly_results

# ==================== EXECUTION ====================
if __name__ == "__main__":
    print("SPE9 Reservoir Optimization System")
    print("Final Version - Production Ready")
    
    print("\nSelect mode:")
    print("1. Quick demo (fast results)")
    print("2. Full interactive system")
    
    mode = input("\nSelect option (1 or 2): ").strip()
    
    if mode == '1':
        env, results = quick_demo()
    else:
        main()
    
    print("\n" + "="*80)
    print("PROGRAM COMPLETED SUCCESSFULLY!")
    print("="*80)
