import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import datetime

# 撲克牌的花色和數字
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.is_face_up = False
    
    def __str__(self):
        if self.is_face_up:
            return f"{self.suit}{self.rank}"
        return "🂠"  # 牌背

class SolitaireEnv:
    def __init__(self):
        self.hidden_cards = {}  # 追踪蓋住的牌的位置
        self.history = []      # 添加歷史記錄
        self.reset()
    
    def reset(self):
        # 初始化一副新牌
        self.deck = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        random.shuffle(self.deck)
        
        # 初始化遊戲區域
        self.tableau = [[] for _ in range(7)]  # 7列主要遊戲區
        self.foundation = [[] for _ in range(4)]  # 4個基礎堆(A開始往上放)
        self.stock = []  # 剩餘的牌堆
        self.waste = []  # 翻開的牌堆

        # 記錄蓋住牌的位置和信息
        self.hidden_cards = {}
        
        # 發牌
        for i in range(7):
            for j in range(i, 7):
                card = self.deck.pop()
                if i == j:  # 最上面的牌朝上
                    card.is_face_up = True
                else:
                    # 記錄蓋住的牌
                    self.hidden_cards[(j, len(self.tableau[j]))] = (card.suit, card.rank)
                self.tableau[j].append(card)
        
        # 剩下的牌放入stock
        self.stock = self.deck
        
        # 保存初始狀態
        self._save_state()
        
        return self._get_state()
    
    def _get_state(self):
        # 計算固定長度的狀態向量
        # tableau: 7列 * 13張牌 * 3特徵 = 273
        # foundation: 4堆 * 3特徵 = 12
        # waste: 1張 * 3特徵 = 3
        # stock: 1特徵 = 1
        # hidden cards: 7 * 13 * 2 = 182
        # 總共: 273 + 12 + 3 + 1 + 182 = 471個特徵
        
        state = np.zeros(471, dtype=np.float32)
        current_idx = 0
        
        # 編碼tableau
        for pile_idx, pile in enumerate(self.tableau):
            for card_idx in range(13):  # 每列最多13張牌
                base_idx = current_idx + card_idx * 3
                if card_idx < len(pile):
                    card = pile[card_idx]
                    if card.is_face_up:
                        state[base_idx] = 1
                        state[base_idx + 1] = SUITS.index(card.suit) / 3
                        state[base_idx + 2] = RANKS.index(card.rank) / 12
                    else:
                        state[base_idx] = 0
                        state[base_idx + 1] = 0
                        state[base_idx + 2] = 0
                else:
                    state[base_idx] = -1
                    state[base_idx + 1] = -1
                    state[base_idx + 2] = -1
            current_idx += 39  # 移動到下一列 (13 * 3 = 39)
        
        # 編碼foundation
        for pile_idx, pile in enumerate(self.foundation):
            base_idx = current_idx + pile_idx * 3
            if pile:
                top_card = pile[-1]
                state[base_idx] = 1
                state[base_idx + 1] = SUITS.index(top_card.suit) / 3
                state[base_idx + 2] = RANKS.index(top_card.rank) / 12
            else:
                state[base_idx] = 0
                state[base_idx + 1] = 0
                state[base_idx + 2] = 0
        current_idx += 12  # 移動到waste區域 (4 * 3 = 12)
        
        # 編碼waste頂牌
        if self.waste:
            top_card = self.waste[-1]
            state[current_idx] = 1
            state[current_idx + 1] = SUITS.index(top_card.suit) / 3
            state[current_idx + 2] = RANKS.index(top_card.rank) / 12
        else:
            state[current_idx] = 0
            state[current_idx + 1] = 0
            state[current_idx + 2] = 0
        current_idx += 3
        
        # 編碼stock是否還有牌
        state[current_idx] = 1 if self.stock else 0
        current_idx += 1
        
        # 編碼hidden cards
        for i in range(7):
            for j in range(13):
                base_idx = current_idx + (i * 13 + j) * 2
                pos = (i, j)
                if pos in self.hidden_cards:
                    suit, rank = self.hidden_cards[pos]
                    state[base_idx] = 1
                    # 計算牌的估計價值
                    value = self._calculate_card_value(suit, rank)
                    state[base_idx + 1] = value
                else:
                    state[base_idx] = 0
                    state[base_idx + 1] = 0
        
        return state
    
    def _is_valid_move(self, card, destination, dest_pile):
        if destination == 'tableau':
            if not dest_pile:  # 空列只能放K
                return card.rank == 'K'
            top_card = dest_pile[-1]
            # 檢查顏色是否相反且數字是否按順序
            is_red = card.suit in ['♥', '♦']
            top_is_red = top_card.suit in ['♥', '♦']
            if is_red == top_is_red:
                return False
            return RANKS.index(card.rank) == RANKS.index(top_card.rank) - 1
            
        elif destination == 'foundation':
            if not dest_pile:  # 空基礎堆只能放A
                return card.rank == 'A'
            top_card = dest_pile[-1]
            # 檢查是否同花色且數字連續
            return (card.suit == top_card.suit and 
                   RANKS.index(card.rank) == RANKS.index(top_card.rank) + 1)
    
    def get_valid_moves(self):
        valid_moves = []
        
        # 從waste到tableau或foundation的移動
        if self.waste:
            card = self.waste[-1]
            # 檢查到tableau的移動
            for i in range(7):
                if self._is_valid_move(card, 'tableau', self.tableau[i]):
                    valid_moves.append(('waste', 'tableau', i))
            # 檢查到foundation的移動
            for i in range(4):
                if self._is_valid_move(card, 'foundation', self.foundation[i]):
                    valid_moves.append(('waste', 'foundation', i))
        
        # 從tableau到tableau或foundation的移動
        for i in range(7):
            if not self.tableau[i]:
                continue
            for j in range(len(self.tableau[i])):
                if not self.tableau[i][j].is_face_up:
                    continue
                card = self.tableau[i][j]
                # 檢查到其他tableau列的移動
                for k in range(7):
                    if k != i and self._is_valid_move(card, 'tableau', self.tableau[k]):
                        valid_moves.append(('tableau', i, j, 'tableau', k))
                # 檢查到foundation的移動
                if j == len(self.tableau[i]) - 1:  # 只能移動頂牌到foundation
                    for k in range(4):
                        if self._is_valid_move(card, 'foundation', self.foundation[k]):
                            valid_moves.append(('tableau', i, j, 'foundation', k))
        
        # 翻stock牌的操作
        if self.stock:
            valid_moves.append(('stock', 'waste'))
        elif self.waste:  # stock空時可以將waste重置
            valid_moves.append(('reset',))
            
        return valid_moves
    
    def _calculate_card_value(self, suit, rank):
        """計算牌的估計價值"""
        # 基礎價值
        base_value = RANKS.index(rank) / 12.0
        
        # 根據當前foundation的情況調整價值
        for pile in self.foundation:
            if pile and pile[-1].suit == suit:
                if RANKS.index(rank) == RANKS.index(pile[-1].rank) + 1:
                    base_value *= 1.5  # 如果是下一張需要的牌,提高價值
                    break
        
        # 根據tableau的情況調整
        for pile in self.tableau:
            if pile and pile[-1].is_face_up:
                top_card = pile[-1]
                if (RANKS.index(rank) == RANKS.index(top_card.rank) - 1 and
                    ((suit in ['♥', '♦']) != (top_card.suit in ['♥', '♦']))):
                    base_value *= 1.2  # 如果可以放在tableau上,稍微提高價值
                    break
        
        # 根據rank額外調整
        if rank == 'A':
            base_value *= 1.3  # A的價值略高，因為是foundation的起始牌
        elif rank == 'K':
            base_value *= 1.2  # K的價值也略高，因為可以開新列
            
        # 如果是紅心或方塊，稍微降低價值（因為需要黑桃或梅花作為基礎）
        if suit in ['♥', '♦']:
            base_value *= 0.9
            
        return base_value

    def _save_state(self):
        """保存當前遊戲狀態"""
        state = {
            'tableau': [[str(card) for card in pile] for pile in self.tableau],
            'foundation': [[str(card) for card in pile] for pile in self.foundation],
            'stock': [str(card) for card in self.stock],
            'waste': [str(card) for card in self.waste],
            'hidden_cards': {str(k): v for k, v in self.hidden_cards.items()}
        }
        self.history.append(state)

    def _update_hidden_cards(self):
        """更新蓋住的牌的信息"""
        new_hidden_cards = {}
        for i, pile in enumerate(self.tableau):
            for j, card in enumerate(pile):
                if not card.is_face_up:
                    new_hidden_cards[(i, j)] = (card.suit, card.rank)
        self.hidden_cards = new_hidden_cards

    def step(self, action):
        # 保存當前狀態用於計算reward
        prev_valid_moves = len(self.get_valid_moves())
        prev_foundation_cards = sum(len(pile) for pile in self.foundation)
        prev_face_up_cards = sum(1 for pile in self.tableau for card in pile if card.is_face_up)
        
        # 保存當前狀態
        self._save_state()
        
        reward = 0
        done = False
        
        if action[0] == 'stock':
            # 從stock翻牌到waste
            card = self.stock.pop()
            card.is_face_up = True
            self.waste.append(card)
            reward = -0.1
            
        elif action[0] == 'reset':
            # 重置waste到stock
            self.stock = list(reversed(self.waste))
            self.waste = []
            for card in self.stock:
                card.is_face_up = False
            reward = -0.2
            
        elif action[0] == 'waste':
            card = self.waste.pop()
            if action[1] == 'tableau':
                self.tableau[action[2]].append(card)
                reward = 1.0
            else:  # foundation
                self.foundation[action[2]].append(card)
                reward = 2.0
                
        elif action[0] == 'tableau':
            source_pile = self.tableau[action[1]]
            cards_to_move = source_pile[action[2]:]
            if action[3] == 'tableau':
                self.tableau[action[4]].extend(cards_to_move)
                del source_pile[action[2]:]
                reward = 0.5
            else:  # foundation
                self.foundation[action[4]].append(cards_to_move[0])
                del source_pile[-1]
                reward = 2.0
            
            # 翻開移動後露出的牌
            if source_pile and not source_pile[-1].is_face_up:
                source_pile[-1].is_face_up = True
                # 更新hidden_cards
                if (action[1], len(source_pile)-1) in self.hidden_cards:
                    del self.hidden_cards[(action[1], len(source_pile)-1)]
                reward += 1.0
        
        # 更新hidden_cards
        self._update_hidden_cards()
        
        # 計算進展獎勵
        current_valid_moves = len(self.get_valid_moves())
        moves_diff = current_valid_moves - prev_valid_moves
        
        if moves_diff > 0:
            reward += moves_diff * 0.2
        
        current_foundation_cards = sum(len(pile) for pile in self.foundation)
        current_face_up_cards = sum(1 for pile in self.tableau for card in pile if card.is_face_up)
        
        # 特殊獎勵：完成一個花色的序列
        for foundation in self.foundation:
            if len(foundation) > 0:
                if foundation[-1].rank == 'K':
                    reward += 5.0
        
        # 檢查是否有連續的foundation牌
        for i in range(4):
            if len(self.foundation[i]) >= 2:
                reward += len(self.foundation[i]) * 0.5

        # 額外的進展獎勵
        foundation_progress = current_foundation_cards - prev_foundation_cards
        face_up_progress = current_face_up_cards - prev_face_up_cards
        
        reward += foundation_progress * 2.0
        reward += face_up_progress * 0.5
        
        # 根據hidden_cards的信息調整獎勵
        for pos, (suit, rank) in self.hidden_cards.items():
            value = self._calculate_card_value(suit, rank)
            if value > 0.7:  # 如果是高價值的牌還被蓋著
                reward -= 0.1  # 輕微懲罰
        
        # 檢查是否獲勝
        if all(len(pile) == 13 for pile in self.foundation):
            reward = 50
            done = True
            
        return self._get_state(), reward, done
    
        # 增加連續成功的獎勵
        if foundation_progress > 0:
            reward += foundation_progress * 3.0  # 增加基礎獎勵
            
        # 增加完成特定目標的獎勵
        if current_foundation_cards >= 13:  # 完成一組
            reward += 10.0
            
        # 懲罰無效的循環
        if moves > 100 and foundation_progress == 0:
            reward -= 0.5 * (moves - 100) / 100

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        # 主幹網絡
        self.fc1 = nn.Linear(input_size, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        
        # 殘差連接
        self.residual1 = nn.Linear(input_size, 512)
        self.residual2 = nn.Linear(512, 256)
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        
        # 輸出層
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, output_size)
        
        # 降低 dropout 率但增加位置
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        # 使用 GELU 激活函數
        self.gelu = nn.GELU()
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, training=False):
        # 主幹網絡前向傳播
        identity = x
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        if training:
            x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        
        # 第一個殘差連接
        residual = self.residual1(identity)
        x = x + residual
        
        if training:
            x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.gelu(x)
        
        # 第二個殘差連接
        residual = self.residual2(residual)
        x = x + residual
        
        # 注意力機制
        # 重塑張量以適應注意力層
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)  # [batch_size, 1, features]
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # [batch_size, features]
        
        if training:
            x = self.dropout3(x)
        
        # 輸出層
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.gelu(x)
        x = self.fc5(x)
        
        return x

    def get_attention_weights(self, x):
        """獲取注意力權重用於可視化"""
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        _, weights = self.attention(x, x, x, need_weights=True)
        return weights

class SolitaireAI:
    def __init__(self):
        self.env = SolitaireEnv()
        self.state_size = 471  # 更新為新的狀態向量大小
        self.action_size = 100
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0003
        self.batch_size = 1024
        self.target_update = 20
        
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1,
            patience=100,
            min_lr=1e-5,
            verbose=True
        )
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # 基礎優先級
        priority = abs(reward) + 1.0
        # 增加優先級計算
        priority = abs(reward) + 1.0
        
        # 根據動作類型調整優先級
        if action[0] == 'tableau' and action[3] == 'foundation':
            priority *= 1.3
        elif action[0] == 'waste' and action[1] == 'foundation':
            priority *= 1.2
        # 根據foundation進展增加優先級
        if action[0] == 'tableau' and action[3] == 'foundation':
            priority *= 2.0  # 增加權重

        # 添加時間衰減因子
        if len(self.memory) > 0:
            oldest_priority = self.memory[0][5]
            priority *= 0.99  # 輕微的時間衰減
        # 增加成功序列的優先級
        if reward > 5:  # 表示有重要進展
            priority *= 1.5
        
        self.memory.append((state, action, reward, next_state, done, priority))
    
    def act(self, state):
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return None
        
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        self.model.eval()  # 設置為評估模式
        with torch.no_grad():
            act_values = self.model(state, training=False)
        self.model.train()  # 恢復為訓練模式
        
        # 將Q值映射到有效移動
        valid_move_values = []
        for move in valid_moves:
            move_idx = self._encode_action(move)
            valid_move_values.append((move, act_values[0][move_idx].item()))
        return max(valid_move_values, key=lambda x: x[1])[0]
    
    def _encode_action(self, action):
        # 將動作編碼為整數索引
        if action[0] == 'stock':
            return 0
        elif action[0] == 'reset':
            return 1
        elif action[0] == 'waste':
            if action[1] == 'tableau':
                return 2 + action[2]
            else:  # foundation
                return 9 + action[2]
        elif action[0] == 'tableau':
            return 13 + action[1] * 7 + action[2]
        return 0
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        
        # 使用溫和的優先級採樣
        priorities = np.array([m[5] for m in self.memory])
        probs = (priorities - priorities.min()) / (priorities.max() - priorities.min() + 1e-7)
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor(np.vstack([m[0] for m in minibatch]))
        actions = torch.LongTensor([self._encode_action(m[1]) for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor(np.vstack([m[3] for m in minibatch]))
        dones = torch.FloatTensor([m[4] for m in minibatch])

        # Double DQN with target network
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        # 計算目標Q值並增加獎勵縮放
        target_q_values = rewards * 0.1 + (1 - dones) * self.gamma * next_q_values
        
        # 計算當前Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 使用 Huber Loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 更新模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 動態學習率調整
        if hasattr(self, 'last_loss') and self.last_loss is not None:
            if loss.item() > self.last_loss * 1.2:  # 更寬鬆的閾值
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95  # 更溫和的衰減
        
        self.last_loss = loss.item()
        
        return loss.item()
    
    def train(self, episodes, checkpoint_interval=None, checkpoint_dir=None):
        """訓練AI玩接龍遊戲
        
        Args:
            episodes (int): 要訓練的回合數
            checkpoint_interval (int, optional): 保存檢查點的間隔回合數
            checkpoint_dir (str, optional): 檢查點保存目錄
        """
        best_reward = float('-inf')
        episode_rewards = []
        best_moves = 0
        no_improvement_count = 0
        episode_loss = 0.0
        loss_count = 0
        
        # 創建檢查點目錄
        if checkpoint_interval and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        for episode in range(episodes):
            # 更新目標網絡
            if episode % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Target network updated at episode {episode}")
                
            # 在每個episode開始時更新epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 如果有足夠的歷史記錄，檢查是否需要增加探索
            if len(episode_rewards) > 100:
                recent_avg = np.mean(episode_rewards[-100:])
                if len(episode_rewards) > 200:
                    previous_avg = np.mean(episode_rewards[-200:-100])
                    if recent_avg < previous_avg * 0.95:
                        self.epsilon = min(self.epsilon * 1.05, 0.8)
                        print(f"Increasing exploration: epsilon = {self.epsilon:.3f}")
            
            state = self.env.reset()
            total_reward = 0
            moves = 0
            last_progress = 0
            episode_loss = 0
            loss_count = 0
            
            while True:
                action = self.act(state)
                if action is None:  # 沒有有效移動
                    break
                    
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                moves += 1
                
                # 計算遊戲進展
                foundation_cards = sum(len(pile) for pile in self.env.foundation)
                if foundation_cards > last_progress:
                    last_progress = foundation_cards
                    moves_limit = 500  # 重置移動限制
                
                # 存儲經驗
                self.remember(state, action, reward, next_state, done)
                
                # 如果有足夠的經驗，進行批量學習
                if len(self.memory) >= self.batch_size:
                    loss = self.replay(self.batch_size)
                    episode_loss += loss
                    loss_count += 1
                
                state = next_state
                
                if done:
                    break
                
                # 動態調整移動限制
                base_moves = 300
                extra_moves = foundation_cards * 20
                tableau_cards = sum(1 for pile in self.env.tableau for card in pile if card.is_face_up)
                extra_moves += tableau_cards * 10
                current_limit = min(800, base_moves + extra_moves)
                
                if moves >= current_limit and foundation_cards == last_progress:
                    break
            
            # episode 結束時更新學習率
            if loss_count > 0:
                avg_loss = episode_loss / loss_count
                self.scheduler.step(avg_loss)
                
            # 記錄並輸出訓練進度
            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # 更新最佳記錄
            if total_reward > best_reward:
                best_reward = total_reward
                best_moves = moves
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 每10局輸出一次訓練信息
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1}")
                print(f"Total Reward: {total_reward}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Best Reward: {best_reward}")
                print(f"Best Moves: {best_moves}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Memory Size: {len(self.memory)}")
                print(f"Moves: {moves}")
                print(f"Foundation Cards: {foundation_cards}")
                print(f"Tableau Face Up Cards: {sum(1 for pile in self.env.tableau for card in pile if card.is_face_up)}")
                print(f"Loss: {self.last_loss:.4f}")
                print("------------------------")
            
            # 保存檢查點
            if checkpoint_interval and checkpoint_dir and (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode+1}.pt")
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # 提前結束條件
            if len(episode_rewards) >= 100 and avg_reward > 95 and no_improvement_count > 200:
                print(f"Training completed early at episode {episode + 1}")
                print(f"Reason: Reached target performance")
                break
            
            if no_improvement_count > 500:
                print(f"Training completed early at episode {episode + 1}")
                print(f"Reason: No improvement for too long")
                break
            # 增加早期停止條件
            patience = 200  # 增加耐心值
            min_delta = 0.01  # 最小改進閾值

            # 動態調整學習率
            if no_improvement_count > 100:
                self.learning_rate *= 0.5
                print(f"Reducing learning rate to {self.learning_rate}")
        
        print("\nTraining finished.")
        print(f"Best reward achieved: {best_reward}")
        print(f"Best moves in a single episode: {best_moves}")
        print(f"Final epsilon value: {self.epsilon:.3f}")
        print(f"Final memory size: {len(self.memory)}")
        return episode_rewards
    
    def save_model(self, filepath):
        """保存模型到文件"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """從文件加載模型"""
        if not os.path.exists(filepath):
            print(f"No model file found at {filepath}")
            return False
            
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint['memory']
        print(f"Model loaded from {filepath}")
        return True

    def evaluate(self, num_episodes=100):
        """評估模型性能"""
        self.model.eval()
        rewards = []
        moves_list = []
        win_count = 0
        foundation_cards_list = []
        
        original_epsilon = self.epsilon
        self.epsilon = 0.01  # 在評估時使用較小的探索率
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            moves = 0
            
            while True:
                action = self.act(state)
                if action is None:
                    break
                    
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                moves += 1
                state = next_state
                
                if done:
                    win_count += 1
                    break
                    
                if moves >= 2000:  # 設置最大移動次數
                    break
            
            rewards.append(total_reward)
            moves_list.append(moves)
            foundation_cards = sum(len(pile) for pile in self.env.foundation)
            foundation_cards_list.append(foundation_cards)
            
            if (episode + 1) % 10 == 0:
                print(f"Evaluation Episode {episode + 1}")
                print(f"Reward: {total_reward:.2f}")
                print(f"Moves: {moves}")
                print(f"Foundation Cards: {foundation_cards}")
                print("------------------------")
        
        self.epsilon = original_epsilon
        self.model.train()
        
        # 計算統計信息
        avg_reward = np.mean(rewards)
        avg_moves = np.mean(moves_list)
        avg_foundation = np.mean(foundation_cards_list)
        win_rate = win_count / num_episodes
        
        print("\nEvaluation Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Moves: {avg_moves:.2f}")
        print(f"Average Foundation Cards: {avg_foundation:.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        
        return {
            'avg_reward': avg_reward,
            'avg_moves': avg_moves,
            'avg_foundation': avg_foundation,
            'win_rate': win_rate
        }

    def visualize_game(self, state):
        """可視化當前遊戲狀態"""
        tableau_str = "Tableau:\n"
        for i, pile in enumerate(self.env.tableau):
            tableau_str += f"{i}: "
            for card in pile:
                tableau_str += f"{str(card)} "
            tableau_str += "\n"
        
        foundation_str = "Foundation:\n"
        for i, pile in enumerate(self.env.foundation):
            foundation_str += f"{i}: "
            for card in pile:
                foundation_str += f"{str(card)} "
            foundation_str += "\n"
        
        waste_str = "Waste: "
        if self.env.waste:
            waste_str += str(self.env.waste[-1])
        
        stock_str = f"Stock: {len(self.env.stock)} cards"
        
        print("\n" + "="*50)
        print(foundation_str)
        print("-"*50)
        print(tableau_str)
        print("-"*50)
        print(waste_str)
        print(stock_str)
        print("="*50 + "\n")

if __name__ == "__main__":
    ai = SolitaireAI()
    
    # 設置保存目錄
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 測試狀態向量
    state = ai.env.reset()
    print("Initial state shape:", state.shape)
    print("Initial state dtype:", state.dtype)
    print("Expected state size:", ai.state_size)
    assert len(state) == ai.state_size, f"State size mismatch: got {len(state)}, expected {ai.state_size}"
    
    # 開始訓練
    try:
        rewards = ai.train(1000, checkpoint_interval=30, checkpoint_dir=f"{save_dir}/checkpoints")
        
        # 保存最終模型
        ai.save_model(f"{save_dir}/final_model.pt")
        
        # 進行最終評估
        print("\nFinal Evaluation:")
        final_eval = ai.evaluate(num_episodes=100)
        
        # 保存評估結果
        with open(f"{save_dir}/evaluation_results.txt", 'w') as f:
            f.write(f"Final Evaluation Results:\n")
            for key, value in final_eval.items():
                f.write(f"{key}: {value}\n")
        
        # 繪製訓練進度圖
        
        # 創建兩個子圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 繪製獎勵曲線
        ax1.plot(rewards)
        ax1.set_title('Training Progress - Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # 繪製移動平均獎勵曲線
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(rewards)), moving_avg)
        ax2.set_title(f'Moving Average Reward (Window Size: {window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
        
        # 計算並輸出詳細的統計信息
        print("\nTraining Statistics:")
        print(f"Total Episodes: {len(rewards)}")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Standard Deviation: {np.std(rewards):.2f}")
        print(f"Maximum Reward: {max(rewards):.2f}")
        print(f"Minimum Reward: {min(rewards):.2f}")
        print(f"Median Reward: {np.median(rewards):.2f}")
        print(f"Last 100 Episodes Average: {np.mean(rewards[-100:]):.2f}")
        
    except Exception as e:
        print(f"Training error: {e}")
        raise