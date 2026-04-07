-- MuscleKing 聊天系统数据库初始化脚本
-- 创建聊天相关的表结构

-- 设置字符集和排序规则
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- 创建聊天会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(255) PRIMARY KEY COMMENT 'Session UUID',
    user_id VARCHAR(255) COMMENT 'User identifier (device ID, anonymous UUID, etc.) - no authentication',
    title VARCHAR(500) NOT NULL COMMENT 'Session title (usually derived from first query)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Session creation timestamp',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Whether the session is active (soft delete flag)',
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='聊天会话表';

-- 创建聊天消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR(255) PRIMARY KEY COMMENT 'Message UUID',
    session_id VARCHAR(255) NOT NULL COMMENT 'Session UUID this message belongs to',
    content TEXT NOT NULL COMMENT 'Message content',
    is_user BOOLEAN DEFAULT TRUE COMMENT 'True if user message, False if assistant message',
    route VARCHAR(100) COMMENT 'Route type for agent processing',
    route_logic TEXT COMMENT 'Route logic explanation',
    metadata JSON COMMENT 'Additional metadata as JSON',
    order_index INTEGER NOT NULL COMMENT 'Order of message in conversation',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Message creation timestamp',
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_session_order (session_id, order_index),
    INDEX idx_created_at (created_at),
    INDEX idx_is_user (is_user)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='聊天消息表';

-- 创建会话状态快照表
CREATE TABLE IF NOT EXISTS chat_session_snapshots (
    id VARCHAR(255) PRIMARY KEY COMMENT 'Snapshot UUID',
    session_id VARCHAR(255) NOT NULL COMMENT 'Session UUID this snapshot belongs to',
    state_data JSON NOT NULL COMMENT 'Session state data as JSON',
    title VARCHAR(500) COMMENT 'Snapshot title or description',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Snapshot creation timestamp',
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话状态快照表';

-- 设置外键检查
SET FOREIGN_KEY_CHECKS = 1;

-- 插入一些测试数据（可选）
INSERT IGNORE INTO chat_sessions (id, user_id, title, created_at, is_active) VALUES
('test-session-001', 'test_user_001', '健身训练咨询', NOW(), TRUE),
('test-session-002', 'test_user_002', '营养计划讨论', NOW(), TRUE);

INSERT IGNORE INTO chat_messages (id, session_id, content, is_user, order_index, created_at) VALUES
('msg-001', 'test-session-001', '我想开始健身训练，应该从哪里开始？', TRUE, 1, NOW()),
('msg-002', 'test-session-001', '很高兴为您提供建议！建议从基础的有氧运动和力量训练开始...', FALSE, 2, NOW()),
('msg-003', 'test-session-002', '我需要制定一个营养计划', TRUE, 1, NOW()),
('msg-004', 'test-session-002', '让我们来设计一个适合您的营养计划...', FALSE, 2, NOW());

-- 显示创建的表
SHOW TABLES LIKE 'chat_%';

-- 显示表结构
DESCRIBE chat_sessions;
DESCRIBE chat_messages;
DESCRIBE chat_session_snapshots;

-- 显示插入的数据
SELECT COUNT(*) as session_count FROM chat_sessions;
SELECT COUNT(*) as message_count FROM chat_messages;
SELECT COUNT(*) as snapshot_count FROM chat_session_snapshots;
