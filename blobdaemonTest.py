#!/usr/bin/env python3
import json
import urllib.request
import urllib.parse
import urllib.error

# 配置
DAEMON_URL = "http://localhost:9998"
TEST_NODE_IP = "/10.0.1.1"  # 根据你的daemon代码，这应该匹配工作目录结构

def send_post_request(url, data):
    """发送POST请求"""
    print(f"发送POST请求到: {url}")
    print(f"请求数据: {json.dumps(data, indent=2)}")
    
    try:
        json_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Content-Length': str(len(json_data))
            }
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            response_data = response.read().decode('utf-8')
            
            print(f"✅ 状态码: {status_code}")
            print(f"📄 响应体: {response_data}")
            
            if status_code == 200:
                try:
                    result = json.loads(response_data)
                    print(f"📊 计算出的总大小: {result.get('sizes', 0)} MiB")
                except json.JSONDecodeError:
                    print("⚠️ 响应不是有效的JSON格式")
            
            return True
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP错误: {e.code} - {e.reason}")
        try:
            error_response = e.read().decode('utf-8')
            print(f"错误响应: {error_response}")
        except:
            pass
        return False
    except urllib.error.URLError as e:
        print(f"❌ 连接错误: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def test_bundles_endpoint():
    """测试 /bundles/ 接口"""
    print("\n" + "="*50)
    print("🧪 测试 /bundles/ 接口")
    print("="*50)
    
    # 根据你的代码，第一个元素是app名称，后面是prefabs
    test_data = [
        {
            "spectype": "image", 
            "name": "test-app",  # 这应该在apps.json中定义
            "specifier": "latest", 
            "size": 100.5
        },
        {
            "spectype": "package",
            "name": "test-bundle",
            "specifier": "v1.0.0", 
            "size": 50.2
        }
    ]
    
    url = f"{DAEMON_URL}/bundles{TEST_NODE_IP}"
    return send_post_request(url, test_data)

def test_layers_endpoint():
    """测试 /layers/ 接口"""
    print("\n" + "="*50)
    print("🧪 测试 /layers/ 接口")
    print("="*50)
    
    # 根据代码，这个接口只使用第一个元素
    test_data = [
        {
            "spectype": "image",
            "name": "testimg1",  # 这应该在payload.json的虚拟manifest中定义
            "specifier": "latest",
            "size": 200.0
        }
    ]
    
    url = f"{DAEMON_URL}/layers{TEST_NODE_IP}"
    return send_post_request(url, test_data)

def check_daemon_connection():
    """检查与daemon的连接"""
    print("="*50)
    print("🔍 检查daemon连接状态")
    print("="*50)
    
    try:
        # 尝试连接daemon，使用一个简单的请求
        req = urllib.request.Request(f"{DAEMON_URL}/")
        with urllib.request.urlopen(req, timeout=3) as response:
            print("✅ Daemon正在运行")
            return True
    except urllib.error.URLError as e:
        if "Connection refused" in str(e):
            print("❌ 无法连接到daemon - 请确保daemon正在运行在9998端口")
        else:
            print(f"⚠️ 连接问题: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 连接检查失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试Daemon HTTP接口...")
    
    # 检查连接
    connected = check_daemon_connection()
    
    # 运行基本测试
    print("\n" + "="*50)
    print("开始API测试...")
    print("="*50)
    
    test_bundles_endpoint()
    test_layers_endpoint()
    
    print("\n🎯 测试完成!")
    if not connected:
        print("\n💡 提示: 请确保你的daemon程序正在运行:")
        print("   go run main.go  # 或者运行你编译的可执行文件")