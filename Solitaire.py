import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os
import sys
import datetime
import copy

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
        """初始化接龍環境"""
        tableau_size = 7 * 13 * 3     # 7列，每列最多13張牌，每張牌3個特徵 = 273
        foundation_size = 4 * 3       # 4個基礎堆，每堆頂部牌3個特徵 = 12
        waste_size = 1 * 3           # 廢牌堆頂部1張牌，3個特徵 = 3
        stock_size = 1               # 股票堆大小信息 = 1
        hidden_cards_size = 52 * 3   # 所有隱藏牌的信息 = 156
        
        self.state_size = 445  # 更新為實際計算出的大小
        
        # 初始化其他屬性
        self.tableau = [[] for _ in range(7)]
        self.foundation = [[] for _ in range(4)]
        self.stock = []
        self.waste = []
        self.move_history = []
        self.hidden_cards = {}
        self.moves_without_progress = 0
        self.history = []

    def _can_move_to_foundation(self, card, foundation_index):
        """檢查牌是否可以移動到指定的foundation堆"""
        dest_pile = self.foundation[foundation_index]
        
        # 如果foundation為空，只能放A
        if not dest_pile:
            return card.rank == 'A'
        
        # 檢查是否同花色且順序正確
        top_card = dest_pile[-1]
        return (card.suit == top_card.suit and 
                RANKS.index(card.rank) == RANKS.index(top_card.rank) + 1)

    def save_decision_point(self):
        """保存當前狀態作為決策點"""
        current_state = GameState()
        current_state.tableau = copy.deepcopy(self.tableau)
        current_state.foundation = copy.deepcopy(self.foundation)
        current_state.stock = copy.deepcopy(self.stock)
        current_state.waste = copy.deepcopy(self.waste)
        current_state.known_cards = copy.deepcopy(self.known_cards)
        self.decision_points.append({
            'state': current_state,
            'moves_tried': set(),
            'position': len(self.state_history)
        })

    def backtrack(self):
        """回溯到上一個決策點"""
        if not self.decision_points:
            return None
            
        last_point = self.decision_points[-1]
        valid_moves = self.get_valid_moves()
        untried_moves = [move for move in valid_moves 
                        if move not in last_point['moves_tried']]
        
        if not untried_moves:
            self.decision_points.pop()
            return self.backtrack()
            
        # 恢復到決策點的狀態
        state = last_point['state']
        self.tableau = copy.deepcopy(state.tableau)
        self.foundation = copy.deepcopy(state.foundation)
        self.stock = copy.deepcopy(state.stock)
        self.waste = copy.deepcopy(state.waste)
        self.known_cards = copy.deepcopy(state.known_cards)
        
        # 選擇一個未嘗試的移動
        next_move = untried_moves[0]
        last_point['moves_tried'].add(next_move)
        
        return next_move

    def remember_card(self, position, card):
        """記錄新發現的牌"""
        self.known_cards[position] = card
        
    def get_known_cards_info(self):
        """獲取已知牌的信息"""
        return self.known_cards

    def is_deadlock(self):
        """檢查是否陷入死局"""
        # 實現死局檢測邏輯
        # 例如：連續多次沒有新的有效移動
        # 或者發現某些關鍵牌被阻塞等情況
        pass
    
    # 在 SolitaireEnv 的 reset 方法中
    def reset(self, custom_deck=None):
        """重置遊戲狀態"""
        # 初始化各個牌堆
        self.tableau = [[] for _ in range(7)]
        self.foundation = [[] for _ in range(4)]
        self.stock = []
        self.waste = []
        self.move_history = []
        self.hidden_cards = {}
        self.moves_without_progress = 0
        
        if custom_deck:
            # 使用自定義牌組
            deck = []
            for suit, rank in custom_deck:
                card = Card(suit, rank)
                deck.append(card)
            
            # 按列發牌（從左到右，每列從上到下）
            card_index = 0
            for col in range(7):  # 7列
                for row in range(col + 1):  # 每列的牌數
                    if card_index >= len(deck):
                        print(f"Warning: Not enough cards in custom deck")
                        break
                    card = deck[card_index]
                    # 只有最後一張牌正面朝上
                    card.is_face_up = (row == col)
                    self.tableau[col].append(card)
                    if not card.is_face_up:
                        self.hidden_cards[(col, row)] = (card.suit, card.rank)
                    card_index += 1
            
            # 剩餘的牌放入stock
            self.stock = deck[card_index:]
            
            # 驗證發牌結果
            print("\nVerifying tableau distribution:")
            for i, pile in enumerate(self.tableau):
                print(f"Column {i} ({len(pile)} cards): ", end="")
                for card in pile:
                    if card.is_face_up:
                        print(f"{card.suit}{card.rank}* ", end="")
                    else:
                        print(f"{card.suit}{card.rank} ", end="")
                print()
            
            print(f"\nStock ({len(self.stock)} cards): ", end="")
            for card in self.stock:
                print(f"{card.suit}{card.rank} ", end="")
            print("\n")
            
        else:
            # 創建並洗牌標準52張牌
            deck = []
            for suit in ['♠', '♥', '♦', '♣']:
                for rank in ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']:
                    card = Card(suit, rank)
                    deck.append(card)
            
            random.shuffle(deck)
            
            # 按列發牌
            card_index = 0
            for col in range(7):
                for row in range(col + 1):
                    card = deck[card_index]
                    card.is_face_up = (row == col)
                    self.tableau[col].append(card)
                    if not card.is_face_up:
                        self.hidden_cards[(col, row)] = (card.suit, card.rank)
                    card_index += 1
            
            # 剩餘的牌放入stock
            self.stock = deck[card_index:]
        
        # 返回初始狀態
        initial_state = self._get_state()
        return initial_state, {}
    
    def _get_state(self):
        """獲取當前遊戲狀態的向量表示"""
        state = []
        
        # Tableau 狀態 (273 = 7 piles * 13 cards * 3 features)
        for pile in self.tableau:
            pile_state = []
            for i in range(13):  # 最多13張牌
                if i < len(pile):
                    card = pile[i]
                    # 每張牌用3個值表示：花色(4種)、點數(13種)、是否朝上
                    pile_state.extend([
                        ['♠', '♥', '♦', '♣'].index(card.suit) / 3,
                        RANKS.index(card.rank) / 12,
                        1.0 if card.is_face_up else 0.0
                    ])
                else:
                    pile_state.extend([0, 0, 0])  # 空位置
            state.extend(pile_state)
        
        # Foundation 狀態 (12 = 4 piles * 3 features)
        for pile in self.foundation:
            if pile:
                top_card = pile[-1]
                state.extend([
                    ['♠', '♥', '♦', '♣'].index(top_card.suit) / 3,
                    RANKS.index(top_card.rank) / 12,
                    len(pile) / 13
                ])
            else:
                state.extend([0, 0, 0])
        
        # Waste 狀態 (3 features)
        if self.waste:
            top_card = self.waste[-1]
            state.extend([
                ['♠', '♥', '♦', '♣'].index(top_card.suit) / 3,
                RANKS.index(top_card.rank) / 12,
                1.0
            ])
        else:
            state.extend([0, 0, 0])
        
        # Stock 狀態 (1 feature)
        state.append(len(self.stock) / 24)  # 標準化
        
        # Hidden cards 信息 (156 = 4 suits * 13 ranks * 3 features)
        hidden_state = np.zeros(156)
        for (pile_idx, card_idx), (suit, rank) in self.hidden_cards.items():
            base_idx = (RANKS.index(rank) * 4 + ['♠', '♥', '♦', '♣'].index(suit)) * 3
            hidden_state[base_idx:base_idx+3] = [
                ['♠', '♥', '♦', '♣'].index(suit) / 3,
                RANKS.index(rank) / 12,
                1.0
            ]
        state.extend(hidden_state)
        
        # 確保狀態向量大小正確
        state_array = np.array(state, dtype=np.float32)
        assert len(state_array) == 445, f"State size mismatch: got {len(state_array)}, expected 445"
        
        return state_array
    
    def _is_valid_move(self, card, dest_type, dest_pile):
        """檢查移動是否有效"""
        if dest_type == 'foundation':
            # 如果目標是foundation
            if not dest_pile:  # 如果foundation為空
                return card.rank == 'A'  # 只能放A
            else:
                # 檢查是否同花色且順序正確
                top_card = dest_pile[-1]
                return (card.suit == top_card.suit and 
                    RANKS.index(card.rank) == RANKS.index(top_card.rank) + 1)
        
        elif dest_type == 'tableau':
            # 如果目標是tableau
            if not dest_pile:  # 如果tableau為空
                return card.rank == 'K'  # 只能放K
            else:
                # 檢查顏色是否相反且順序正確
                top_card = dest_pile[-1]
                return (self._is_opposite_color(card.suit, top_card.suit) and 
                    RANKS.index(card.rank) == RANKS.index(top_card.rank) - 1)
        
        return False

    def _is_opposite_color(self, suit1, suit2):
        """檢查兩個花色是否顏色相反"""
        return ((suit1 in ['♥', '♦'] and suit2 in ['♠', '♣']) or
                (suit1 in ['♠', '♣'] and suit2 in ['♥', '♦']))
    
    def get_valid_moves(self):
        valid_moves = []
        
        # 檢查tableau到foundation的移動
        for i, source_pile in enumerate(self.tableau):
            if source_pile:
                card = source_pile[-1]
                if card.is_face_up:
                    for j in range(4):  # 檢查所有foundation堆
                        if self._can_move_to_foundation(card, j):
                            valid_moves.append(('tableau', i, len(source_pile)-1, 'foundation', j))

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
        value = 0.0
        # 基礎價值
        base_value = RANKS.index(rank) / 12.0
        
        # 檢查是否可以直接放到foundation
        for i, pile in enumerate(self.foundation):
            if not pile and rank == 'A':
                value += 1.0
            elif pile and pile[-1].suit == suit:
                if RANKS.index(rank) == RANKS.index(pile[-1].rank) + 1:
                    value += 1.0
        
        # A和2的基礎價值較高
        if rank in ['A', '2']:
            value += 0.5

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

        rank_values = {
            'A': 1.0,
            'K': 0.85,
            'Q': 0.7,
            'J': 0.55,
            '10': 0.4,
            '9': 0.35,
            '8': 0.3,
            '7': 0.25,
            '6': 0.2,
            '5': 0.15,
            '4': 0.1,
            '3': 0.05,
            '2': 0.0
        }
        
        base_value = rank_values[rank]   
        return base_value

    def _save_state(self):
        """保存當前遊戲狀態"""
        state = {
            'tableau': copy.deepcopy(self.tableau),
            'foundation': copy.deepcopy(self.foundation),
            'stock': copy.deepcopy(self.stock),
            'waste': copy.deepcopy(self.waste),
            'hidden_cards': copy.deepcopy(self.hidden_cards),
            'moves_without_progress': self.moves_without_progress,
            'move_history': copy.deepcopy(self.move_history)
        }
        self.history.append(state)

    def _restore_state(self):
        """恢復到上一個遊戲狀態"""
        if not self.history:
            return False
            
        state = self.history.pop()
        self.tableau = state['tableau']
        self.foundation = state['foundation']
        self.stock = state['stock']
        self.waste = state['waste']
        self.hidden_cards = state['hidden_cards']
        self.moves_without_progress = state['moves_without_progress']
        self.move_history = state['move_history']
        return True

    def _update_hidden_cards(self):
        """更新蓋住的牌的信息"""
        new_hidden_cards = {}
        for i, pile in enumerate(self.tableau):
            for j, card in enumerate(pile):
                if not card.is_face_up:
                    new_hidden_cards[(i, j)] = (card.suit, card.rank)
        self.hidden_cards = new_hidden_cards

    def step(self, action):
        """執行一個動作並返回新狀態、獎勵和是否結束"""
        # 保存當前狀態
        self._save_state()
        prev_valid_moves = len(self.get_valid_moves())
        prev_foundation_cards = sum(len(pile) for pile in self.foundation)
        prev_face_up_cards = sum(1 for pile in self.tableau for card in pile if card.is_face_up)
        
        # 初始化獎勵和結束標誌
        reward = 0
        done = False
        
        if action[0] == 'stock':
            # 從stock翻牌到waste
            if self.stock:
                card = self.stock.pop()
                card.is_face_up = True
                self.waste.append(card)
                reward = 0.1  # 小獎勵
            else:
                # 如果stock空了，但waste有牌，重置
                if self.waste:
                    self.stock = self.waste[::-1]  # 反轉waste
                    for card in self.stock:
                        card.is_face_up = False
                    self.waste = []
                    reward = -0.2  # 小懲罰
                else:
                    reward = -0.5  # 較大懲罰

        if action[0] == 'tableau' and action[3] == 'foundation':
            # 從tableau移到foundation
            source_pile = self.tableau[action[1]]
            if len(source_pile) > action[2]:  # 確保有足夠的牌
                card = source_pile[action[2]]
                if card.is_face_up:  # 確保牌是朝上的
                    dest_pile = self.foundation[action[4]]
                    if self._is_valid_move(card, 'foundation', dest_pile):
                        # 移動牌
                        dest_pile.append(source_pile.pop(action[2]))
                        reward = 2.0  # 成功移到foundation的獎勵
                        
                        # 如果移動後露出新牌，翻開它
                        if source_pile and not source_pile[-1].is_face_up:
                            source_pile[-1].is_face_up = True
                            if (action[1], len(source_pile)-1) in self.hidden_cards:
                                del self.hidden_cards[(action[1], len(source_pile)-1)]
                            reward += 1.0
                    else:
                        reward = -1.0  # 無效移動的懲罰
            
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
                # 檢查是否是無意義的移動
                if self._is_meaningless_move(source_pile, action[2], self.tableau[action[4]]):
                    reward = -1.0  # 加大懲罰
                else:
                    reward = 0.5
                    
                self.tableau[action[4]].extend(cards_to_move)
                del source_pile[action[2]:]
            
            # 翻開移動後露出的牌
            if source_pile and not source_pile[-1].is_face_up:
                source_pile[-1].is_face_up = True
                # 更新hidden_cards
                if (action[1], len(source_pile)-1) in self.hidden_cards:
                    del self.hidden_cards[(action[1], len(source_pile)-1)]
                reward += 1.0
        
        # 更新hidden_cards
        self._update_hidden_cards()
        
        # 保持歷史記錄在合理範圍內
        if len(self.move_history) > 100:
            self.move_history.pop(0)

        # 計算進展獎勵
        current_valid_moves = len(self.get_valid_moves())
        moves_diff = current_valid_moves - prev_valid_moves
        
        if moves_diff > 0:
            reward += moves_diff * 0.2
        
        current_foundation_cards = sum(len(pile) for pile in self.foundation)
        current_face_up_cards = sum(1 for pile in self.tableau for card in pile if card.is_face_up)
                
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
            
        # 增加連續成功的獎勵
        if foundation_progress > 0:
            reward += foundation_progress * 3.0  # 增加基礎獎勵
            # 重置無效循環計數器
            self.moves_without_progress = 0
        
        else:
            # 增加無效循環計數器
            self.moves_without_progress += 1
            
        # 懲罰無效循環
        if self.moves_without_progress > 20:  # 如果連續20步沒有進展
            reward -= 0.1 * (self.moves_without_progress - 20)  # 逐步增加懲罰
            
        if self.moves_without_progress > 50:  # 如果連續50步沒有進展
            reward -= 0.2 * (self.moves_without_progress - 50)  # 進一步增加懲罰
                                    
        # 檢查是否獲勝
        if all(len(pile) == 13 for pile in self.foundation):
            reward = 50
            done = True
        
        # 獲取新狀態
        new_state = self._get_state()

        return new_state, reward, done

    def _is_useful_card(self, card):
        """檢查一張牌是否當前有用"""
        # 檢查是否可以直接放到foundation
        for pile in self.foundation:
            if self._is_valid_move(card, 'foundation', pile):
                return True
                
        # 檢查是否可以放到tableau並能幫助解鎖其他牌
        for i, pile in enumerate(self.tableau):
            if self._is_valid_move(card, 'tableau', pile):
                # 如果這個位置下面有蓋著的牌,則更有價值
                if pile and not pile[-1].is_face_up:
                    return True
                # 如果這張牌能幫助解鎖其他有用的序列
                if self._will_unlock_useful_sequence(card, i):
                    return True
        
        return False

    def _will_unlock_useful_sequence(self, card, tableau_idx):
        """檢查這張牌是否能幫助解鎖有用的序列"""
        # 例如:如果這張牌是紅心5,檢查是否有黑桃4或梅花4等待它
        target_rank = RANKS[RANKS.index(card.rank) - 1]
        target_suits = ['♠', '♣'] if card.suit in ['♥', '♦'] else ['♥', '♦']
        
        for pile in self.tableau:
            if pile and pile[-1].is_face_up:
                if (pile[-1].rank == target_rank and 
                    pile[-1].suit in target_suits):
                    return True
        return False
    
    def _is_meaningless_move(self, source_pile, start_idx, dest_pile):
        """檢查是否是無意義的移動"""
        # 如果目標堆是空的,但移動的不是K,就是無意義的
        if not dest_pile and source_pile[start_idx].rank != 'K':
            return True
            
        # 如果移動後會露出新牌,則不是無意義的
        if start_idx > 0 and not source_pile[start_idx-1].is_face_up:
            return False
            
        # 如果這個移動會讓其他有用的移動成為可能,則不是無意義的
        if self._enables_useful_moves(source_pile[start_idx:], dest_pile):
            return False
            
        # 檢查是否只是在兩堆之間來回移動
        if len(self.move_history) >= 4:
            last_moves = self.move_history[-4:]
            if self._is_cycling_moves(last_moves):
                return True
                
        return False

    def _enables_useful_moves(self, cards_to_move, dest_pile):
        """檢查這個移動是否能使其他有用的移動成為可能"""
        # 例如:移動後能釋放出一個空列來放K
        # 或者能讓某張被擋住的牌可以移動到foundation
        return False  # 實現具體的檢查邏輯

    def _is_cycling_moves(self, moves):
        """檢查是否在進行循環移動"""
        # 檢查最近的移動是否形成了循環模式
        if len(moves) < 4:
            return False
            
        # 檢查是否在同樣的位置之間來回移動
        if (moves[0][1] == moves[2][4] and moves[0][4] == moves[2][1] and
            moves[1][1] == moves[3][4] and moves[1][4] == moves[3][1]):
            return True
            
        return False

    def _is_important_decision(self, action):
        """判斷是否是重要決策點"""
        if action[0] == 'tableau':
            # 移動King到空列
            if (len(self.tableau[action[1]]) > 0 and 
                self.tableau[action[1]][action[2]].rank == 'K' and 
                not self.tableau[action[4]]):
                return True
            # 移動多張牌
            if action[2] < len(self.tableau[action[1]]) - 1:
                return True
        elif action[0] == 'waste':
            # 從waste移動到tableau的重要牌
            card = self.waste[-1]
            if card.rank in ['A', 'K'] or self._is_blocking_card(card):
                return True
        return False

    def _is_blocking_card(self, card):
        """判斷一張牌是否是阻塞其他重要操作的關鍵牌"""
        # 實現檢測邏輯
        pass

    def choose_backtrack_point(self):
        """智能選擇回溯點"""
        if not self.decision_points:
            return None
            
        # 評估每個決策點
        best_point = None
        best_score = float('-inf')
        
        for point in reversed(self.decision_points):
            score = self._evaluate_decision_point(point)
            if score > best_score:
                best_score = score
                best_point = point
                
        return best_point

    def _evaluate_decision_point(self, point):
        """評估決策點的價值"""
        state = point['state']
        score = 0
        
        # 根據foundation的進度
        foundation_cards = sum(len(pile) for pile in state.foundation)
        score += foundation_cards * 2
        
        # 根據已知牌的信息
        known_cards = len(state.known_cards)
        score += known_cards
        
        # 根據tableau中面朝上的牌
        face_up_cards = sum(1 for pile in state.tableau 
                        for card in pile if card.is_face_up)
        score += face_up_cards
        
        # 根據剩餘的未嘗試移動數
        untried_moves = len(self.get_valid_moves()) - len(point['moves_tried'])
        score += untried_moves * 0.5
        
        return score

class GameState:
    def __init__(self):
        self.tableau = []
        self.foundation = []
        self.stock = []
        self.waste = []
        self.known_cards = {}  # 記錄已知的牌
        self.move_history = []  # 移動歷史
        self.decision_points = []  # 重要決策點
        self.score = 0
        self.moves_made = 0

    def clone(self):
        """創建當前狀態的深度複製"""
        new_state = GameState()
        new_state.tableau = copy.deepcopy(self.tableau)
        new_state.foundation = copy.deepcopy(self.foundation)
        new_state.stock = copy.deepcopy(self.stock)
        new_state.waste = copy.deepcopy(self.waste)
        new_state.known_cards = copy.deepcopy(self.known_cards)
        new_state.move_history = copy.deepcopy(self.move_history)
        new_state.score = self.score
        new_state.moves_made = self.moves_made
        return new_state

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        # 分離特徵提取
        self.tableau_net = nn.Sequential(
            nn.Linear(273, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.foundation_net = nn.Sequential(
            nn.Linear(12, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.waste_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        self.hidden_cards_net = nn.Sequential(
            nn.Linear(156, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 合併層
        self.combine = nn.Sequential(
            nn.Linear(256 + 64 + 32 + 128 + 1, 512),  # +1 for stock
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        # 分離輸入
        tableau = x[:, :273]
        foundation = x[:, 273:285]
        waste = x[:, 285:288]
        stock = x[:, 288:289]
        hidden_cards = x[:, 289:]
        
        # 處理各部分
        t_out = self.tableau_net(tableau)
        f_out = self.foundation_net(foundation)
        w_out = self.waste_net(waste)
        h_out = self.hidden_cards_net(hidden_cards)
        
        # 合併
        combined = torch.cat([t_out, f_out, w_out, stock, h_out], dim=1)
        return self.combine(combined)

class SolitaireAI:
    def __init__(self, custom_deck=None):
        self.env = SolitaireEnv()
        self.custom_deck = custom_deck  
        # 更新狀態大小的計算
        self.state_size = (
            273 +  # Tableau (7 piles * 13 cards * 3 features)
            12 +   # Foundation (4 piles * 3 features)
            3 +    # Waste (3 features)
            1 +    # Stock (1 feature)
            156    # Hidden cards (4 suits * 13 ranks * 3 features)
        )
        self.action_size = 52
        
        # 初始化主網絡和目標網絡
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        
        # 新的學習參數
        self.learning_rate = 0.0005
        self.min_learning_rate = 1e-5
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 使用余弦退火學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,  # 初始周期
            T_mult=2,  # 周期倍增因子
            eta_min=self.min_learning_rate
        )
        
        # 修改探索參數
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.2
        self.gamma = 0.95
        
        # 分離正面和負面經驗
        self.positive_memory = deque(maxlen=5000)
        self.negative_memory = deque(maxlen=5000)
        
        self.batch_size = 64
        self.target_update = 20
        self.env = SolitaireEnv()
        
        # 用於追蹤上一次的損失
        self.last_loss = None

    def calculate_priority(self, reward, done, action):
        """計算經驗的優先級
        
        Args:
            reward (float): 獲得的獎勵
            done (bool): 是否完成遊戲
            action (tuple): 執行的動作
            
        Returns:
            float: 經驗的優先級值
        """
        # 基礎優先級為獎勵的絕對值
        priority = abs(reward) + 1.0
        
        # 特殊情況的優先級提升
        if done and reward > 0:  # 成功完成遊戲
            priority *= 2.0
        
        if action[0] == 'tableau':
            if action[3] == 'foundation':  # 移到foundation的動作
                priority *= 1.5
            elif len(action) > 2:  # 移動多張牌
                priority *= 1.2
        
        elif action[0] == 'waste':
            if action[1] == 'foundation':  # waste到foundation
                priority *= 1.5
            elif action[1] == 'tableau':   # waste到tableau
                priority *= 1.2
        
        # 根據獎勵的正負調整優先級
        if reward > 0:
            priority *= 1.2  # 提高正面經驗的優先級
        elif reward < -0.5:
            priority *= 1.1  # 稍微提高大的負面經驗的優先級
        
        return priority

    def remember(self, state, action, reward, next_state, done):
        """存儲經驗到記憶中"""
        # 確保狀態是正確的格式
        if isinstance(state, tuple):
            state = state[0]  # 如果是元組，取第一個元素（狀態向量）
        if isinstance(next_state, tuple):
            next_state = next_state[0]
            
        experience = (state, action, reward, next_state, done)
        priority = self.calculate_priority(reward, done, action)
        
        if reward > 0:
            self.positive_memory.append((experience, priority))
        else:
            self.negative_memory.append((experience, priority))

    def act(self, state):
        """選擇動作"""
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return None
        
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        # 確保狀態是正確的格式
        if isinstance(state, tuple):
            state = state[0]
            
        state = torch.FloatTensor(state).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        
        # 將Q值映射到有效移動
        valid_move_values = []
        for move in valid_moves:
            move_idx = self._encode_action(move)
            if move_idx >= self.action_size:
                print(f"Warning: action index {move_idx} exceeds action size {self.action_size}")
                continue
            valid_move_values.append((move, act_values[0][move_idx].item()))
        
        if not valid_move_values:
            return random.choice(valid_moves)
        
        return max(valid_move_values, key=lambda x: x[1])[0]
    
    def _encode_action(self, action):
        """將動作編碼為整數索引，確保範圍在 0-51 之間"""
        if action[0] == 'stock':
            return 0
        elif action[0] == 'reset':
            return 1
        elif action[0] == 'waste':
            if action[1] == 'tableau':
                return 2 + min(action[2], 6)  # 2-8 (7個tableau位置)
            else:  # foundation
                return 9 + min(action[2], 3)  # 9-12 (4個foundation位置)
        elif action[0] == 'tableau':
            source_pile = min(action[1], 6)  # 0-6
            card_index = min(action[2], 12)   # 0-12
            
            if action[3] == 'tableau':
                # 13-38: tableau到tableau的移動
                dest_index = min(action[4], 6)
                return min(13 + (source_pile * 4 + dest_index), 51)
            else:  # foundation
                # 39-51: tableau到foundation的移動
                dest_index = min(action[4], 3)
                return min(39 + (source_pile * 2 + dest_index), 51)
        
        return 0  # 默認情況
    
    def replay(self, batch_size):
        """從經驗記憶中採樣並學習"""
        # 確保有足夠的經驗可以採樣
        total_experiences = len(self.positive_memory) + len(self.negative_memory)
        if total_experiences < batch_size:
            return 0.0
        
        # 根據可用經驗動態調整批次大小
        pos_size = min(batch_size // 2, len(self.positive_memory))
        neg_size = min(batch_size - pos_size, len(self.negative_memory))
        actual_batch_size = pos_size + neg_size
        
        if actual_batch_size == 0:
            return 0.0
        
        minibatch = []
        
        # 採樣正面經驗
        if pos_size > 0:
            pos_experiences = [x[0] for x in self.positive_memory]
            pos_priorities = np.array([x[1] for x in self.positive_memory])
            pos_probs = pos_priorities / pos_priorities.sum()
            pos_indices = np.random.choice(len(self.positive_memory), pos_size, p=pos_probs)
            pos_batch = [self.positive_memory[i][0] for i in pos_indices]
            minibatch.extend(pos_batch)
        
        # 採樣負面經驗
        if neg_size > 0:
            neg_experiences = [x[0] for x in self.negative_memory]
            neg_priorities = np.array([x[1] for x in self.negative_memory])
            neg_probs = neg_priorities / neg_priorities.sum()
            neg_indices = np.random.choice(len(self.negative_memory), neg_size, p=neg_probs)
            neg_batch = [self.negative_memory[i][0] for i in neg_indices]
            minibatch.extend(neg_batch)
        
        if not minibatch:  # 如果沒有足夠的樣本
            return 0.0
            
        # 準備批次數據
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
        self.scheduler.step()  # 更新學習率
        
        return loss.item()
    
    def train(self, episodes, checkpoint_interval=None, checkpoint_dir=None):
        # 確保檢查點目錄存在
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # 在方法開始處添加日誌設置
        log_file = os.path.join(checkpoint_dir, "training_log.txt") if checkpoint_dir else "training_log.txt"

        # 確保日誌文件的目錄存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 創建一個日誌記錄器,同時輸出到控制台和文件
        class TeeLogger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'w', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
                
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        
        # 保存原始的標準輸出
        original_stdout = sys.stdout
        sys.stdout = TeeLogger(log_file)
        
        try:
            # 記錄訓練開始時間和基本信息
            start_time = datetime.datetime.now()
            print(f"Training started at: {start_time}")
            print(f"Training configuration:")
            print(f"Episodes: {episodes}")
            print(f"Checkpoint interval: {checkpoint_interval}")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Initial epsilon: {self.epsilon}")
            print(f"Learning rate: {self.learning_rate}")
            print("=" * 50)             
    
            best_reward = float('-inf')
            episode_rewards = []
            best_moves = 0
            no_improvement_count = 0
            prev_avg_reward = float('-inf')
            
            # 創建檢查點目錄
            if checkpoint_interval and checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

            # 添加遊戲記錄功能
            record_episode = 0
            print(f"Will record episode {record_episode}")

            for episode in range(episodes):
                # 重置環境並獲取初始狀態
                state = self.env.reset(self.custom_deck)
                total_reward = 0
                moves = 0
                episode_loss = 0.0
                loss_count = 0
                last_progress = 0
                
                # 當前回合的記錄初始化
                current_episode_record = []
                is_recording = (episode == record_episode)
                
                if is_recording:
                    print("\nStarting recorded episode...")
                    self.visualize_game(state)  # 顯示初始狀態
                
                # 更新目標網絡
                if episode % self.target_update == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                    print(f"Target network updated at episode {episode}")
                
                # 動態調整探索率
                if len(episode_rewards) > 100:
                    current_avg = np.mean(episode_rewards[-100:])
                    if episode > 100:
                        if current_avg < prev_avg_reward * 0.95:
                            self.epsilon = min(self.epsilon * 1.1, 0.9)
                            #print(f"Increasing exploration: epsilon = {self.epsilon:.3f}")
                        else:
                            self.epsilon = max(self.epsilon * 0.995, self.epsilon_min)
                    prev_avg_reward = current_avg

                while True:
                    action = self.act(state)
                    if action is None:  # 沒有有效移動
                        break
                        
                    next_state, reward, done = self.env.step(action)
                    
                    # 記錄遊戲過程
                    if is_recording:
                        step_record = {
                            'move_number': moves + 1,
                            'action': action,
                            'reward': reward,
                            'foundation_cards': sum(len(pile) for pile in self.env.foundation),
                            'face_up_cards': sum(1 for pile in self.env.tableau for card in pile if card.is_face_up)
                        }
                        current_episode_record.append(step_record)
                        
                        # 輸出當前動作和狀態
                        print(f"\nMove {moves + 1}:")
                        print(f"Action: {self._format_action(action)}")
                        print(f"Reward: {reward:.2f}")
                        self.visualize_game(next_state)
                    
                    total_reward += reward
                    moves += 1
                    
                    # 計算遊戲進展
                    foundation_cards = sum(len(pile) for pile in self.env.foundation)
                    if foundation_cards > last_progress:
                        last_progress = foundation_cards
                        moves_limit = 500  # 重置移動限制
                    
                    # 存儲經驗
                    self.remember(state, action, reward, next_state, done)
                    
                    # 檢查總經驗數量並進行訓練
                    total_memories = len(self.positive_memory) + len(self.negative_memory)
                    if total_memories >= self.batch_size:
                        loss = self.replay(self.batch_size)
                        episode_loss += loss
                        loss_count += 1
                        self.last_loss = loss  # 更新最後的損失值
                    
                    state = next_state
                    
                    if done:
                        if is_recording:
                            print("\nGame completed!")
                        break
                    
                    # 動態調整移動限制
                    base_moves = 300
                    extra_moves = foundation_cards * 20
                    tableau_cards = sum(1 for pile in self.env.tableau for card in pile if card.is_face_up)
                    extra_moves += tableau_cards * 10
                    current_limit = min(800, base_moves + extra_moves)
                    
                    if moves >= current_limit and foundation_cards == last_progress:
                        break
                
                # 保存回合記錄
                if is_recording and checkpoint_dir:
                    record_path = os.path.join(checkpoint_dir, f"game_record_episode_{episode}.txt")
                    self._save_game_record(current_episode_record, record_path, total_reward, moves)
                    print(f"\nGame record saved to {record_path}")
                
                # episode 結束時更新學習率
                if loss_count > 0:
                    avg_loss = episode_loss / loss_count
                    self.scheduler.step()
                
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
                    print(f"\nEpisode {episode + 1} Summary:")
                    print(f"Total Reward: {total_reward:.2f}")
                    print(f"Average Reward (last 100): {avg_reward:.2f}")
                    print(f"Best Reward: {best_reward:.2f}")
                    print(f"Moves Made: {moves}")
                    print(f"Foundation Cards: {foundation_cards}")
                    print(f"Face Up Cards: {tableau_cards}")
                    print(f"Current Epsilon: {self.epsilon:.3f}")
                    print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"Positive Memory Size: {len(self.positive_memory)}")
                    print(f"Negative Memory Size: {len(self.negative_memory)}")
                    if self.last_loss is not None:
                        print(f"Latest Loss: {self.last_loss:.4f}")
                    print("-" * 50)
                
                # 保存檢查點
                if checkpoint_interval and checkpoint_dir and (episode + 1) % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode+1}.pt")
                    self.save_model(checkpoint_path)
                    print(f"\nCheckpoint saved at episode {episode + 1}")
                    print(f"Path: {checkpoint_path}")
                    print("-" * 50)
                
                # 提前結束條件
                if len(episode_rewards) >= 100 and avg_reward > 95 and no_improvement_count > 200:
                    print(f"Training completed early at episode {episode + 1}")
                    print(f"Reason: Reached target performance")
                    break
                
                if no_improvement_count > 500:
                    print(f"Training completed early at episode {episode + 1}")
                    print(f"Reason: No improvement for too long")
                    break
                
                # 動態調整學習率
                if no_improvement_count > 100:
                    self.learning_rate *= 0.5
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
                    #print(f"Reducing learning rate to {self.learning_rate}")
            
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print("\n=== Training Summary ===")
            print(f"Training completed at: {end_time}")
            print(f"Total training time: {duration}")
            print(f"\nFinal Results:")
            print(f"Best reward achieved: {best_reward}")
            print(f"Best moves in a single episode: {best_moves}")
            print(f"Final epsilon value: {self.epsilon:.3f}")
            print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Final positive memory size: {len(self.positive_memory)}")
            print(f"Final negative memory size: {len(self.negative_memory)}")
            print(f"\nStatistics:")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(f"Standard deviation: {np.std(episode_rewards):.2f}")
            print(f"Maximum reward: {max(episode_rewards):.2f}")
            print(f"Minimum reward: {min(episode_rewards):.2f}")
            print(f"Median reward: {np.median(episode_rewards):.2f}")
            print(f"Last 100 episodes average: {np.mean(episode_rewards[-100:]):.2f}")
            print("="*50)
            
            return episode_rewards
    
        except Exception as e:
            print(f"\nTraining Error: {str(e)}")
            raise

        finally:
            sys.stdout = original_stdout
    
    def _format_action(self, action):
        """將動作轉換為可讀的字符串"""
        if action[0] == 'stock':
            return "Draw card from stock to waste"
        elif action[0] == 'reset':
            return "Reset waste to stock"
        elif action[0] == 'waste':
            dest = "tableau" if action[1] == 'tableau' else "foundation"
            return f"Move card from waste to {dest} pile {action[2]}"
        elif action[0] == 'tableau':
            source_pile = action[1]
            card_index = action[2]
            dest = "tableau" if action[3] == 'tableau' else "foundation"
            dest_pile = action[4]
            return f"Move card(s) from tableau {source_pile} (position {card_index}) to {dest} pile {dest_pile}"

    def _save_game_record(self, record, filepath, total_reward, total_moves):
        """保存遊戲記錄到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== Solitaire Game Record ===\n\n")
            f.write(f"Total Moves: {total_moves}\n")
            f.write(f"Total Reward: {total_reward:.2f}\n\n")
            
            for step in record:
                f.write(f"Move {step['move_number']}:\n")
                f.write(f"Action: {self._format_action(step['action'])}\n")
                f.write(f"Reward: {step['reward']:.2f}\n")
                f.write(f"Foundation Cards: {step['foundation_cards']}\n")
                f.write(f"Face Up Cards: {step['face_up_cards']}\n")
                f.write("-" * 50 + "\n")

    def save_model(self, filepath):
        """保存模型到文件"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'positive_memory': self.positive_memory,
            'negative_memory': self.negative_memory
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
        self.positive_memory = checkpoint['positive_memory']
        self.negative_memory = checkpoint['negative_memory']
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
        
        for episode in range(episodes):
            # 重置環境並獲取初始狀態
            state, _ = self.env.reset(self.custom_deck)  # 解包返回值
            total_reward = 0
            moves = 0
            
            while True:
                action = self.act(state)
                if action is None:  # 沒有有效移動
                    break
                    
                next_state, reward, done = self.env.step(action)
                
                # 存儲經驗
                self.remember(state, action, reward, next_state, done)
                
                # 更新狀態
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
    # 選擇是否使用自定義牌組
    use_custom_deck = True  # 改為False則使用隨機牌組

    if use_custom_deck:
        # 定義自定義牌組
        custom_deck = [
            # 第一列的牌（從上到下）
            ('♠', 'Q'),
            # 第二列的牌
            ('♣', '7'), ('♣', '6'),
            # 第三列的牌
            ('♣', 'J'), ('♥', 'K'), ('♣', 'Q'),
            # 第四列的牌
            ('♦', '5'), ('♦', '6'), ('♠', '2'), ('♣', '5'),
            # 第五列的牌
            ('♥', '5'), ('♣', '2'), ('♠', '5'), ('♥', '9'), ('♠', 'J'),
            # 第六列的牌
            ('♣', 'K'), ('♣', '8'), ('♥', '6'), ('♣', '10'), ('♣', '9'), ('♥', '7'),
            # 第七列的牌
            ('♠', '6'), ('♦', '8'), ('♦', '4'), ('♠', '3'), ('♦', '2'), ('♣', '4'), ('♦', 'J'),
            # stock中的牌（剩餘的牌）            
            ('♠', '9'), ('♦', '7'), ('♦', 'K'), ('♥', '2'), ('♦', 'Q'),
            ('♣', '3'), ('♥', '8'), ('♠', '7'), ('♠', 'K'), ('♥', 'Q'),
            ('♠', '10'), ('♦', 'A'), ('♠', '8'), ('♥', '10'), ('♠', 'A'),
            ('♦', '9'), ('♥', '3'), ('♥', 'J'), ('♥', 'A'), ('♠', '4'),
            ('♦', '3'), ('♦', '10'), ('♥', '4'), ('♣', 'A'),
        ]
    else:   
        # 創建環境和AI實例
        env = SolitaireEnv()
        ai = SolitaireAI(custom_deck=custom_deck)
        
        # 重置環境，使用自定義牌組
        initial_state, _ = env.reset(custom_deck)
        
        # 驗證初始狀態
        print("\nInitial Game State:")
    
    # 顯示初始狀態
    ai.visualize_game(initial_state)

    # 設置保存目錄
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/{timestamp}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    
    # 創建必要的目錄
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 將狀態轉換為numpy數組以便查看形狀
    state_array = np.array(initial_state)
    print("Initial state shape:", state_array.shape)
    print("Initial state dtype:", state_array.dtype)
    print("Expected state size:", ai.state_size)
    print("Actual state size:", len(state_array))
    
    # 檢查狀態向量的組成
    print("\nState vector composition:")
    print(f"Tableau section (0-272): {state_array[:273].shape}")
    print(f"Foundation section (273-284): {state_array[273:285].shape}")
    print(f"Waste section (285-287): {state_array[285:288].shape}")
    print(f"Stock section (288): {state_array[288]}")
    print(f"Hidden cards section (289-444): {state_array[289:].shape}")
    
    assert len(state_array) == ai.state_size, f"State size mismatch: got {len(state_array)}, expected {ai.state_size}"
    
    # 開始訓練
    try:
        rewards = ai.train(1000, checkpoint_interval=50, checkpoint_dir=checkpoint_dir)
        
        # 保存最終模型
        ai.save_model(os.path.join(save_dir, "final_model.pt"))
        
        '''# 進行最終評估
        print("\nFinal Evaluation:")
        final_eval = ai.evaluate(num_episodes=100)
        
        # 保存評估結果
        with open(f"{save_dir}/evaluation_results.txt", 'w') as f:
            f.write(f"Final Evaluation Results:\n")
            for key, value in final_eval.items():
                f.write(f"{key}: {value}\n")
        
        # 繪製訓練進度圖
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
        plt.savefig(os.path.join(save_dir, 'training_progress.png'))
        plt.close()
        
        # 計算並輸出詳細的統計信息
        print("\nTraining Statistics:")
        print(f"Total Episodes: {len(rewards)}")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Standard Deviation: {np.std(rewards):.2f}")
        print(f"Maximum Reward: {max(rewards):.2f}")
        print(f"Minimum Reward: {min(rewards):.2f}")
        print(f"Median Reward: {np.median(rewards):.2f}")
        print(f"Last 100 Episodes Average: {np.mean(rewards[-100:]):.2f}")'''
        
    except Exception as e:
        print(f"Training error: {e}")
        raise
