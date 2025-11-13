package com.ui.pages.contact;

import com.ui.pages.user.LoginPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ContactTest {
    private WebDriver driver;
    private ContactPage contactPage;
    private MessagesPage messagesPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        LoginPage loginPage = new LoginPage(driver);
        loginPage.loginAs("customer3@practicesoftwaretesting.com", "pass123");
        contactPage = new ContactPage(driver);
        messagesPage = new MessagesPage(driver);
    }

    @Test
    void shouldSendMessage() {
        String uniqueUUID = java.util.UUID.randomUUID().toString();
        String message = uniqueUUID + "text with 50 letters abcdefghijklmnoprstuvwxyzABCD";
        contactPage.open();
        contactPage.selectSubject("Payments");
        contactPage.enterMessage(message);
        contactPage.sendMessage();

        messagesPage.open();
        System.out.println(messagesPage.getMessage(message));
        assertTrue(messagesPage.getMessage(message).contains(uniqueUUID), "The message content does not match");
        assertEquals("NEW", messagesPage.getStatus(message), "The message status is not 'NEW'");
        assertTrue("payments".contains(messagesPage.getSubject(message)), "The message subject is not 'payments'");
    }


    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
