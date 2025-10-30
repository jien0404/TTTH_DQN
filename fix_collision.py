#!/usr/bin/env python3
"""
Script sửa lỗi robot đi qua khe hẹp
"""

import sys
import os

def fix_robot_radius():
    """Tăng robot radius để tránh đi qua khe hẹp"""
    main_py_path = '/Users/fuongfotfet/Downloads/RL Paper/DQL/DQN-main/main.py'
    
    # Đọc file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Thay đổi radius từ 8 → 12
    old_line = 'self.robot = Robot(self.start[0], self.start[1], cell_size, self.controller, vision=cell_size*2.5, radius=8, env_padding=env_padding)'
    new_line = 'self.robot = Robot(self.start[0], self.start[1], cell_size, self.controller, vision=cell_size*2.5, radius=12, env_padding=env_padding)'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Ghi lại file
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        print("✅ Đã tăng robot radius từ 8 → 12 pixels")
        print("🎯 Robot sẽ không thể đi qua khe hẹp < 24 pixels")
        return True
    else:
        print("❌ Không tìm thấy dòng code cần sửa")
        return False

if __name__ == "__main__":
    success = fix_robot_radius()
    if success:
        print("\n🚀 Hãy test lại chương trình để xem robot có còn đi qua khe hẹp không!")
    else:
        print("\n⚠️  Vui lòng sửa thủ công trong main.py, line 173")
