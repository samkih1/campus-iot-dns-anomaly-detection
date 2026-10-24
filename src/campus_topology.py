from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.cli import CLI
from mininet.log import setLogLevel
import time


class CampusIoTTopo(Topo):

    def build(self):

        s1 = self.addSwitch('s1')

        camera = self.addHost('camera', ip='10.0.0.11/24')
        sensor = self.addHost('sensor', ip='10.0.0.12/24')
        classroom = self.addHost('classroom', ip='10.0.0.13/24')
        academic = self.addHost('academic', ip='10.0.0.14/24')

        dns = self.addHost('dns', ip='10.0.0.53/24')
        external = self.addHost('external', ip='10.0.0.100/24')

        self.addLink(camera, s1)
        self.addLink(sensor, s1)
        self.addLink(classroom, s1)
        self.addLink(academic, s1)
        self.addLink(dns, s1)
        self.addLink(external, s1)


def run():

    topo = CampusIoTTopo()

    net = Mininet(
        topo=topo,
        switch=OVSBridge,
        controller=None
    )

    net.start()

    dns = net.get('dns')

    print("\n*** Starting Campus DNS Resolver")

    dns.cmd('rm -f /tmp/campus-dns.log')
    dns.cmd('rm -f /tmp/campus-dnsmasq.pid')

    dns.cmd(
        'dnsmasq '
        '--conf-file=configs/dnsmasq-campus.conf '
        '--pid-file=/tmp/campus-dnsmasq.pid'
    )

    time.sleep(2)

    print("*** Checking DNS resolver")
    result = dns.cmd("ss -lunp | grep '10.0.0.53:53'")
    print(result)

    print("*** Testing topology connectivity")
    net.pingAll()

    print("\n*** Campus IoT DNS research environment ready")
    print("*** DNS Resolver: 10.0.0.53")
    print("*** Entering Mininet CLI\n")

    CLI(net)

    print("\n*** Stopping Campus DNS Resolver")

    dns.cmd(
        'test -f /tmp/campus-dnsmasq.pid && '
        'kill $(cat /tmp/campus-dnsmasq.pid) 2>/dev/null'
    )

    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()
