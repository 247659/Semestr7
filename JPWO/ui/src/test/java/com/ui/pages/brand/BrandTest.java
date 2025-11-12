package com.ui.pages.brand;

import com.ui.pages.user.LoginPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class BrandTest {

    private WebDriver driver;
    private LoginPage loginPage;
    private AddBrandPage addBrandPage;
    private BrandAdminPage brandAdminPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAsAdmin("admin@practicesoftwaretesting.com", "welcome01");
        addBrandPage = new AddBrandPage(driver);
        brandAdminPage = new BrandAdminPage(driver);
    }

    @Test
    void shouldAddNewBrand() {
        String uniqueBrandName = "New Brand " + System.currentTimeMillis();
        String uniqueSlug = "new-brand-" + System.currentTimeMillis();
        addBrandPage.open();
        addBrandPage.fillForm(uniqueBrandName, uniqueSlug);
        brandAdminPage.open();
        brandAdminPage.search(uniqueBrandName);
        String addedBrandName = brandAdminPage.getBrandName(uniqueBrandName);
        String addedBrandSlug = brandAdminPage.getBrandSlug(uniqueBrandName);
        assertEquals(uniqueBrandName, addedBrandName, "Brand name does not match");
        assertEquals(uniqueSlug, addedBrandSlug, "Brand slug does not match");
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
