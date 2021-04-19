package ai.certifai.training.classification.IrdologyClassification;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public class myFrame extends JFrame implements ActionListener {

    JButton btn_upload;
    JLabel label;
    JLabel ClassOutput;

    public static void main(String[] args) {
        new myFrame();
    }
    public myFrame() {


        ImageIcon icon2 = new ImageIcon("C:\\Users\\Skymind\\Desktop\\cdle-traininglabs\\dl4j-cv-labs\\src\\main\\resources\\mockup.png");

        label = new JLabel();
        label.setIcon(icon2);
        label.setBounds(0, 0, icon2.getIconWidth(), icon2.getIconHeight());
        label.setVisible(false);

        btn_upload = new JButton();
        btn_upload.setBounds((icon2.getIconWidth()/2)-100, icon2.getIconHeight()+10, 100, 75);
        btn_upload.addActionListener(this);
        btn_upload.setText("UPLOAD");
        btn_upload.setFocusable(false);
        btn_upload.setHorizontalTextPosition(JButton.CENTER);
        btn_upload.setVerticalTextPosition(JButton.CENTER);
        btn_upload.setFont(new Font("Comic Sans",Font.PLAIN,15));
        btn_upload.setForeground(Color.cyan);
        btn_upload.setBackground(Color.BLACK);
        btn_upload.setBorder(BorderFactory.createEtchedBorder());

        this.setTitle("Tutorial");
        this.setVisible(true);
        this.setSize(icon2.getIconWidth(), icon2.getIconHeight()+150);
        this.setLayout(null);
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        this.add(btn_upload);
        this.add(label);

    }


    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        //Limit type of file name extensions supported.
        FileNameExtensionFilter filter = new FileNameExtensionFilter("4 Extensions Supported", "jpg", "png", "jpeg", "gif");
        fileChooser.setFileFilter(filter);
        int selected = fileChooser.showOpenDialog(null);
        if (selected == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            //Get Path of the selected image.
            String getselectedImage = file.getAbsolutePath();
            //Display image path on Message Dialog
            JOptionPane.showMessageDialog(null, getselectedImage);
            ImageIcon imIco = new ImageIcon(getselectedImage);
            //make image fit on jlabel.
            Image imFit = imIco.getImage();
            Image imgFit = imFit.getScaledInstance(label.getWidth(), label.getHeight(), Image.SCALE_SMOOTH);
            label.setIcon(new ImageIcon(imgFit));

        }
        label.setVisible(true);
        btn_upload.setEnabled(false);

    }
}


